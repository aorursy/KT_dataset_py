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



df_train = pd.read_csv(PATH/'train.csv')

df_test = pd.read_csv(PATH/'test.csv')



# if is_interactive():

#     df_train = df_train.sample(800)



_ = df_train.hist()
# create image data bunch

aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])

data = ImageDataBunch.from_df(df=df_train,

                              path=PATH, folder='train_images', suffix='.png',

                              valid_pct=0.1,

                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.1, max_zoom=1.15, max_rotate=45.),

                              size=224,

                              bs=32, 

                              num_workers=os.cpu_count()

                             ).normalize(aptos19_stats)
# check classes

print(f'Classes: {data.classes}')
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

    model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes':5})



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

model
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

learn.lr_find(start_lr=1e-5, end_lr=1e1, wd=5e-3)

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, max_lr=3e-3, pct_start=0.1, div_factor=10, final_div=30, wd=5e-3, moms=(0.9, 0.8))

learn.save('stage-1')

learn.recorder.plot_losses()
# unfreeze and search appropriate learning rate for full training

learn.unfreeze()

learn.lr_find(start_lr=slice(1e-6, 1e-5), end_lr=slice(1e-2, 1e-1), wd=1e-3)

learn.recorder.plot(suggestion=True)
# train all layers

learn.fit_one_cycle(3, max_lr=slice(1e-4, 1e-3), div_factor=50, final_div=100, wd=1e-3, moms=(0.9, 0.8))

learn.save('stage-2')

learn.recorder.plot_losses()

# schedule of the lr (left) and momentum (right) that the 1cycle policy uses

learn.recorder.plot_lr(show_moms=True)
# _ = learn.load('best_kappa')



# learn.lr_find(start_lr=slice(1e-7, 1e-6), end_lr=slice(1e-2, 1e-1), wd=1e-3)

# learn.recorder.plot(suggestion=True)
# train all layers

learn.fit_one_cycle(cyc_len=25, max_lr=slice(1e-4, 1e-3), pct_start=0, final_div=1000, wd=1e-3, moms=(0.9, 0.8)) # warm restart: pct_start=0

learn.save('stage-3')

learn.recorder.plot_losses()

# # schedule of the lr (left) and momentum (right) that the 1cycle policy uses

learn.recorder.plot_lr(show_moms=True)
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
#move models back to root folder

!mv {learn.model_dir}/*.pth .

os.listdir()