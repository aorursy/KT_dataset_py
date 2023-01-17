import time

t_start = time.time()
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

from fastai.callbacks import *
from functools import partial

from sklearn import metrics

from collections import Counter
import os

import PIL

import cv2
pwd
print(os.listdir("../input/"))
print(os.listdir("../input/aptos2019-blindness-detection"))
sys.path.insert(0, '../input/aptos2019-blindness-detection')

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
dat_path = Path('../input/aptos2019-blindness-detection')

trn_path = Path('../input/aptos2019-blindness-detection/train_images')

(dat_path,trn_path)
train_df = pd.read_csv(dat_path/'train.csv')

train_df.shape
train_df.head()
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

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img



def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
IMG_SIZE = 512

def _load_format(path, convert_mode, after_open)->Image:

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0), 10) ,-4 ,128)

                    

    return Image(pil2tensor(image, np.float32).div_(255)) #return fastai Image format



vision.data.open_image = _load_format
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)
bs = 64
data = ImageDataBunch.from_df(path=dat_path, folder='train_images', df=train_df, suffix='.png', 

                               fn_col=0, label_col=1,valid_pct=0.2, size=256, ds_tfms=tfms).normalize(imagenet_stats)
#data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
#from sklearn.metrics import cohen_kappa_score

#def quadratic_kappa(y_hat, y):

#    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa])
learn.fit_one_cycle(1)
learn.fit_one_cycle(1)

learn.unfreeze()

learn.fit_one_cycle(9, max_lr=slice(1e-4,1e-3))
#learn.unfreeze()

#learn.fit_one_cycle(1, max_lr=slice(1e-4,1e-3))
learn.model_dir = Path('/kaggle/working/')
learn.save('/kaggle/working/FastAI_APTOS_epoch_11')
test_df = pd.read_csv(dat_path/'sample_submission.csv')
test_df.head()
learn.data.add_test(ImageList.from_df(test_df,dat_path,folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
test_df.diagnosis = preds.argmax(1)

test_df.head()
test_df.to_csv('submission.csv',index=False)
t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))