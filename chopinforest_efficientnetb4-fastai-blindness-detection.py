import time

# For keeping time. GPU limit for this competition is set to ± 9 hours.

t_start = time.time()





# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid', color_codes=True)



from sklearn.metrics import confusion_matrix



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm, tqdm_notebook

import os, random

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

from sklearn.utils import shuffle



!ls ../input/*
import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from fastai.basic_train import *

from fastai.vision.learner import *



fastai.__version__
# check if the kernel is running in interactive/edit/debug mode: https://www.kaggle.com/masterscrat/detect-if-notebook-is-running-interactively

def is_interactive():

   return 'runtime' in get_ipython().config.IPKernelApp.connection_file



print('Interactive?', is_interactive())
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(42)
# copy pretrained weights to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

model_path = '/tmp/.cache/torch/checkpoints/efficientNet.pth'

!cp ../input/efficientnet*/efficientNet_*.pth {model_path}
PATH = Path('../input/aptos2019-blindness-detection')

PATH_train = Path('../input/drd-newold/drd')



df_train = pd.read_csv('../input/oldandnew/new_train_data.csv')

df_test = pd.read_csv(PATH/'test.csv')



# if is_interactive():

#     df_train = df_train.sample(800)



_ = df_train.hist()
SIZE=380

def crop_image_from_gray(img, tol=7):

    """

    Applies masks to the orignal image and 

    returns the a preprocessed image with 

    3 channels

    """

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

def preprocess_image(image):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,10), -4, 128)

    return image



def open_aptos2019_image(fn, convert_mode, after_open)->Image:

    image = cv2.imread(fn)

    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (SIZE, SIZE))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,10), -4, 128)

    return Image(pil2tensor(image, np.float32).div_(255))



vision.data.open_image = open_aptos2019_image
# create image data bunch

aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])

data = ImageDataBunch.from_df(df=df_train,

                              path=PATH_train, folder='aptos_drd_jpeg', suffix='.jpeg',

                              valid_pct=0.1,

                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.05, max_rotate=20.),

                              #size=380,

                              bs=4, 

                              num_workers=os.cpu_count()

                             ).normalize(aptos19_stats)
# check classes

print(f'Classes: \n {data.classes}')
# show some sample images

data.show_batch(rows=3, figsize=(7,6))
package_path = '../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master'

sys.path.append(package_path)



from efficientnet_pytorch import EfficientNet
# FastAI adapators to retrain our model without lossing its old head ;)

def EfficientNetB4(pretrained=True):

    """Constructs a EfficientNet model for FastAI.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 5 })



    if pretrained:

        model_state = torch.load(model_path)

        # load original weights apart from its head

        if '_fc.weight' in model_state.keys():

            model_state.pop('_fc.weight')

            model_state.pop('_fc.bias')

            res = model.load_state_dict(model_state, strict=False)

            assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'

        else:

            # A basic remapping is required

            from collections import OrderedDict

            mapping = { i:o for i,o in zip(model_state.keys(), model.state_dict().keys()) }

            mapped_model_state = OrderedDict([

                (mapping[k], v) for k,v in model_state.items() if not mapping[k].startswith('_fc')

            ])

            res = model.load_state_dict(mapped_model_state, strict=False)

            print(res)

    return model
# create model

model = EfficientNetB4(pretrained=True)

# print model structure (hidden)

#model
class FocalLoss(nn.Module):

    def __init__(self, gamma=3., reduction='mean'):

        super().__init__()

        self.gamma = gamma

        self.reduction = reduction



    def forward(self, inputs, targets):

        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)

        F_loss = ((1 - pt)**self.gamma) * CE_loss

        if self.reduction == 'sum':

            return F_loss.sum()

        elif self.reduction == 'mean':

            return F_loss.mean()
# from FastAI master

from torch.utils.data.sampler import WeightedRandomSampler



class OverSamplingCallback(LearnerCallback):

    def __init__(self,learn:Learner, weights:torch.Tensor=None):

        super().__init__(learn)

        self.labels = self.learn.data.train_dl.dataset.y.items

        _, counts = np.unique(self.labels, return_counts=True)

        self.weights = (weights if weights is not None else

                        torch.DoubleTensor((1/counts)[self.labels]))



    def on_train_begin(self, **kwargs):

        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(

            WeightedRandomSampler(self.weights, len(self.learn.data.train_dl.dataset)),

            self.learn.data.train_dl.batch_size,False)
# build model (using EfficientNet)

learn = Learner(data, model,

                loss_func=FocalLoss(),

                metrics=[accuracy, KappaScore(weights="quadratic")],

                callback_fns=[BnFreeze,

#                               OverSamplingCallback,

#                               partial(GradientClipping, clip=0.2),

                              partial(SaveModelCallback, monitor='kappa_score', name='best_kappa')]

               )

learn.split( lambda m: (model._conv_head,) )

learn.freeze()

learn.model_dir = '/tmp/'
# train head first

learn.freeze()

learn.lr_find(start_lr=1e-6, end_lr=1e1, wd=5e-3)

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=3e-2, div_factor=10, final_div=100, wd=5e-3)

learn.save('stage-1')

learn.recorder.plot_losses()
# unfreeze and search appropriate learning rate for full training

learn.unfreeze()

#learn.lr_find(start_lr=slice(1e-6, 1e-5), end_lr=slice(1e-2, 1e-1), wd=1e-3)

#learn.recorder.plot(suggestion=True)
# train all layers

learn.fit_one_cycle(3, max_lr=slice(1e-4, 1e-3), div_factor=10, final_div=100, wd=1e-3)

learn.save('stage-2')

#learn.recorder.plot_losses()

# schedule of the lr (left) and momentum (right) that the 1cycle policy uses

#learn.recorder.plot_lr(show_moms=True)
# _ = learn.load('best_kappa')



# learn.lr_find(start_lr=slice(1e-7, 1e-6), end_lr=slice(1e-2, 1e-1), wd=1e-3)

# learn.recorder.plot(suggestion=True)
# train all layers

learn.fit_one_cycle(cyc_len=5, max_lr=slice(5e-5, 5e-4), pct_start=0, wd=1e-3) # warm restart: pct_start=0

learn.save('stage-3')

#learn.recorder.plot_losses()

# # schedule of the lr (left) and momentum (right) that the 1cycle policy uses

#learn.recorder.plot_lr(show_moms=True)
# learn.load('best_kappa')



# # retrain only head

# learn.freeze()

# learn.lr_find(start_lr=1e-7, end_lr=1e-1, wd=1e-2)

# learn.recorder.plot(suggestion=True)
# learn.fit_one_cycle(6, max_lr=1e-3, div_factor=100, wd=1e-2)

# learn.save('stage-4')
learn.load('best_kappa')



interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
# interp.plot_top_losses(5, figsize=(15,11))  ## TODO: fix loss function reduction topk
# remove zoom from FastAI TTA

tta_params = {'beta':0.12, 'scale':1.0}
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(

    sample_df, PATH,

    folder='test_images',

    suffix='.png'

))
preds,y = learn.TTA(ds_type=DatasetType.Test, **tta_params)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)

_ = sample_df.hist()
sample_df['diagnosis'].value_counts()
#move models back to root folder

!mv {learn.model_dir}/*.pth .

os.listdir()
# Check kernels run-time. GPU limit for this competition is set to ± 9 hours.

t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))