# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from shutil import copyfile

from fastai.vision import *

import cv2

from sklearn.metrics import cohen_kappa_score

import scipy as sp





# Any results you write to the current directory are saved as output.
def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
random_seed(101, True)
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet101/resnet101.pth' '/tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'
# from fastai_slack import read_webhook_url, SlackCallback

# hook = 'https://hooks.slack.com/services/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'

# slack_cb = SlackCallback('surbhi', hook, frequency=1)
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')



x_train = df_train['id_code']

y_train = df_train['diagnosis']
# SIZE=224



# train_df=pd.read_csv(PATH+'/train.csv')

# test_df=pd.read_csv(PATH+'/sample_submission.csv'
PATH = "../input/aptos2019-blindness-detection/"



train = ImageList.from_df(df_train, path=PATH, cols='id_code', folder="train_images", suffix='.png')

test = ImageList.from_df(df_test, path=PATH, cols='id_code', folder="test_images", suffix='.png')
y_train.value_counts()
def crop_image(img,tol=7):        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def open_aptos2019_image(fn, convert_mode, after_open)->Image:

    image = cv2.imread(fn)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = crop_image(image)

    image = cv2.resize(image, (224, 224))

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image , (0,0) , 128/10) ,-4 ,128)

    return Image(pil2tensor(image, np.float32).div_(255))



vision.data.open_image = open_aptos2019_image

#np.random.seed(42)

tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)

data = (

    train.split_by_rand_pct(0.2)

    .label_from_df(cols='diagnosis', label_cls=FloatList)

    .add_test(test)

    .transform(tfms, size=224)

    .databunch(path=Path('.'), bs=32).normalize(imagenet_stats)

)
data.show_batch(rows=3, figsize=(7,6))

def qk(y_pred, y):

    #y_pred = torch.argmax(y_pred, 1)

    return torch.tensor(cohen_kappa_score(torch.round(y_pred.float()), y, weights='quadratic'),device='cuda:0')
learn = cnn_learner(data, models.resnet101, metrics=[qk], model_dir=".", callback_fns=ShowGraph)

# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(10, 1e-2, callbacks=[callbacks.SaveModelCallback(learn,monitor='qk',mode='max', name='best_model')])

learn.load('best_model')
learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4), callbacks=[callbacks.SaveModelCallback(learn,monitor='qk',mode='max', name='best_model')])

#learn.load('best_model')
#learn.save('resnet_50_stage-1')
## Progressive Resizing of the Image
# data = (

#     train.split_by_rand_pct(0.2)

#     .label_from_df(cols='diagnosis', label_cls=FloatList)

#     .add_test(test)

#     .transform(tfms, size=256)

#     .databunch(path=Path('.'), bs=32).normalize(imagenet_stats)

# )
# learn = cnn_learner(data, models.resnet50, metrics=[qk], model_dir=".", callback_fns=ShowGraph)

#learn.load('resnet_50_stage-1')

# learn.load('best_model')
#learn.lr_find()

#learn.recorder.plot()
# learn.fit_one_cycle(10, 1e-3, callbacks=[callbacks.SaveModelCallback(learn,monitor='qk',mode='max', name='best_model')])

# learn.load('best_model')
# learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(10, max_lr=slice(3e-6,2e-3), callbacks=[callbacks.SaveModelCallback(learn,monitor='qk',mode='max')])
learn.load('best_model')
valid_preds, valid_y = learn.TTA(ds_type=DatasetType.Valid)

test_preds, _ = learn.TTA(ds_type=DatasetType.Test)
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(valid_preds, valid_y)

coefficients = optR.coefficients()



valid_predictions = optR.predict(valid_preds, coefficients)[:,0].astype(int)

test_predictions = optR.predict(test_preds, coefficients)[:,0].astype(int)



valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")
valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")
print("coefficients:", coefficients)

print("validation score:", valid_score)
df_test.diagnosis = test_predictions

df_test.to_csv("submission.csv", index=None)

df_test.head()