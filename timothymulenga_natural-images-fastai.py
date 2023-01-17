import warnings
warnings.filterwarnings('ignore')

#setting up our enviroment
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
p