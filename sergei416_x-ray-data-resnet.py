# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import os

import matplotlib.image as mpimg

from sklearn.metrics import cohen_kappa_score

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

import json

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight, shuffle

import cv2





from fastai.core import *

from fastai.basic_data import *

from fastai.basic_train import *

from fastai.torch_core import *

IMG_SIZE = 512
# Making pretrained weights work without needing to find the default filename

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')

!cp '/kaggle/input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
os.listdir('../input')
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head(10)
len_df = len(df)

print(f"There are {len_df} images.")
#histogram data visualization

df['diagnosis'].hist(figsize = (10, 5))




for i in df['path'][0:3]:

    img = mpimg.imread(i)





    imgplot = plt.imshow(img)

    plt.show()

    

 



from PIL import Image



for i in range(3):

    im = Image.open(df['path'][i])

    width, height = im.size

    print(width,height) 

bs = 64 

sz=224
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=SEED)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=SEED)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

train_y.hist()

valid_y.hist()


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)

src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset

        .split_by_rand_pct(0.2) #Splitting the dataset

        .label_from_df(cols='diagnosis',label_cls=FloatList) #obtain labels from the level column

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation

        .databunch(bs=bs,num_workers=4) #DataBunch

        .normalize(imagenet_stats) #Normalize     

       )

###
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

data.show_batch(rows=3, figsize=(7,6))
def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
learn = cnn_learner(data, base_arch=models.resnet50, metrics = [quadratic_kappa])
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,max_lr = 1e-2)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.export()

learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
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



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



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

optR.fit(valid_preds[0],valid_preds[1])
coefficients = optR.coefficients()

print(coefficients)
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.tail()
learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))
preds,y = learn.TTA(ds_type=DatasetType.Test)
test_predictions = optR.predict(preds, coefficients)
sample_df.diagnosis = test_predictions.astype(int)

sample_df.tail()
sample_df.to_csv('submission.csv',index=False)