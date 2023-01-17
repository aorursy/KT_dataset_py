# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# This file contains all the main external libs we'll use
from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import pandas as pd
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH+'valid')
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])
arch=resnet34
data=ImageClassifierData.from_paths(path=PATH,val_name='valid',test_name='test',tfms=tfms_from_model(arch, sz))
data.classes
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 2)
log_preds = learn.predict(is_test=True)
lps = np.argmax(log_preds,axis=1)
lps.shape
preds_classes = [data.classes[i] for i in lps]
sub_frame = pd.read_csv('../input/Sample_Sub.csv')
sub_frame['Category'] = preds_classes
sub_frame.head()
sub_frame.to_csv('Submission.csv',index=False)
