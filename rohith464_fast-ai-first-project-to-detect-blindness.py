%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 42

seed_everything(SEED)
print(torch.cuda.is_available())
import os

path = '../input/aptos2019-blindness-detection/'

train_csv_path = path +'train.csv'

train_img_path = path + 'train_images/'

train = pd.read_csv(train_csv_path)



#test path strings

print(train_csv_path)

print(train_img_path)
train.head()
print("There are total {} images in training dataset".format(len(train)))
f_names = get_image_files(train_img_path)

f_names[:3]
train['id_code'] = train['id_code'].map(lambda x: (train_img_path + x + '.png'))



#test the paths

print(train['id_code'][1])

print(train['id_code'][2])
tfms = get_transforms(do_flip=True,flip_vert=True,

                      max_rotate=360,max_warp=0,max_zoom=1.1,

                      max_lighting=0.1,p_lighting=0.5)
data = ImageDataBunch.from_df(path = '', df= train, label_col='diagnosis', ds_tfms=tfms,

                              valid_pct=0.2, size=224, bs=32).normalize(imagenet_stats)
import PIL

import cv2

IMG_SIZE = 224
from PIL import Image



def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



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

            #print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

        #print(img.shape)

        return img
from fastai.vision import Image



def _load_format(path,convert_mode, after_open) -> Image :

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), 10) ,-4 ,128)

    img_fastai = Image((pil2tensor(image, np.float32)).div_(255)) #fastai Image format

    

    return (img_fastai)



vision.data.open_image = _load_format
data.show_batch(rows = 3, figsize= (12,10))
from sklearn.metrics import confusion_matrix



def quadratic_kappa(actuals, preds):

    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition

    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 

    of adoption rating."""

    w = np.zeros((5,5))

    O = confusion_matrix(actuals, preds)

    for i in range(5): 

        for j in range(5):

            w[i][j] = float(((i-j)**2)/16)

    

    act_hist=np.zeros([5])

    for item in actuals: 

        act_hist[item]+=1

    

    pred_hist=np.zeros([5])

    for item in preds: 

        pred_hist[item]+=1

    

    E = np.outer(act_hist, pred_hist);

    E = E/E.sum();

    O = O/O.sum();

    

    num=0

    den=0

    for i in range(5):

        for j in range(5):

            num+=w[i][j]*O[i][j]

            den+=w[i][j]*E[i][j]

    return (1 - (num/den))
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet34, metrics=[error_rate,kappa])
learn.model
learn.fit_one_cycle(4)
learn.save('Stage-1')
interpret = ClassificationInterpretation.from_learner(learn)

losses,idx = interpret.top_losses()



interpret.plot_top_losses(9, figsize=(15,11))
interpret.plot_confusion_matrix(figsize=(12,12), dpi=60)
interpret.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('Stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-06,1e-03))
sample_df = pd.read_csv(path+'/sample_submission.csv')

sample_df.head()
data.add_test(ImageList.from_df(sample_df,path,folder='test_images',suffix='.png'))
data.show_batch(rows=1)
# remove zoom from FastAI TTA

tta_params = {'beta':0.12, 'scale':1.0}
learn.data.add_test(ImageList.from_df(sample_df,path,folder='test_images',suffix='.png'))
preds,y = learn.TTA(ds_type=DatasetType.Test, **tta_params)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)