%reload_ext autoreload
%autoreload 2
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = '../input/'
sz = 224
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH)
os.listdir(f'{PATH}valid')
files = os.listdir(f'{PATH}valid/cats')[:5]
files
img = plt.imread(f'{PATH}valid/cats/{files[0]}')
plt.imshow(img)
img.shape
img[:4, :4]
shutil.rmtree(f'{PATH}tmp', ignore_errors=True)
arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)





