# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install --upgrade transformers
!pip install simpletransformers
# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU

GPUs = GPU.getGPUs()
gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()
import numpy as np
import pandas as pd
#from google.colab import files
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import gc
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import sklearn
from sklearn.metrics import log_loss
from sklearn.metrics import *
from sklearn.model_selection import *
import re
import random
import torch
pd.options.display.max_colwidth = 200

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

seed_all(2)
train=pd.read_csv('/kaggle/input/train-data/Train_BNBR.csv')
test=pd.read_csv('/kaggle/input/health-data/Test_health.csv')
sub = pd.read_csv('/kaggle/input/health-data/ss_health.csv')
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder
train['label'] = train['label'].astype('category')  
train['label'] = train['label'].cat.codes
train.head()
train.label.value_counts()
print(train['text'].apply(lambda x: len(x.split())).describe())
print(test['text'].astype(str).apply(lambda x: len(x)).describe())
train1=train.drop(['ID'],axis=1)
test1=test.drop(['ID'],axis=1)
test1['target']=0
import math
#%%writefile setup.sh

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#!sh setup.sh
%%time
err=[]
y_pred_tot=[]

fold=StratifiedKFold(n_splits=20, shuffle=True, random_state= 2)
i=1
for train_index, test_index in fold.split(train1,train1['label']):
    train1_trn, train1_val = train1.iloc[train_index], train1.iloc[test_index]
    model = ClassificationModel('roberta', 'roberta-base', use_cuda= False,num_labels= 4, args={'train_batch_size':32,
                                                                         'reprocess_input_data': True,
                                                                         'overwrite_output_dir': True,
                                                                         "fp16": False,
                                                                         #"fp16_opt_level": "O2",
                                                                         'do_lower_case': True,
                                                                         'num_train_epochs': 4,
                                                                         'max_seq_length': 128,
                                                                         #"adam_epsilon": 1e-4,
                                                                         #"warmup_ratio": 0.06,
                                                                         'regression': False,
                                                                         #'lr_rate_decay': 0.4,
                                                                         'manual_seed': 2,
                                                                         "learning_rate":1e-4,
                                                                         'weight_decay':0,
                                                                         "save_eval_checkpoints": False,
                                                                         "save_model_every_epoch": False,
                                                                         "silent": True})
    model.train_model(train1_trn)
    raw_outputs_val = model.eval_model(train1_val)[1]
    raw_outputs_val = softmax(raw_outputs_val,axis=1)   #[:,1]  
    print(f"Log_Loss: {log_loss(train1_val['label'], raw_outputs_val)}")
    err.append(log_loss(train1_val['label'], raw_outputs_val)) 
    raw_outputs_test = model.eval_model(test1)[1]
    raw_outputs_test = softmax(raw_outputs_test,axis=1)  #[:,1]  
    y_pred_tot.append(raw_outputs_test)
print("Mean LogLoss: ",np.mean(err))
final=np.mean(y_pred_tot, 0)
final= pd.DataFrame(final)
final.head()
# Current best Mean LogLoss:  0.3973383729609846 (20fold_rbb_1_1e4_wd_0_32_128_epoch4_true), LB = 0.3726
final.to_csv('20fold_rbb_1_1e4_wd_0_32_128_epoch4_clean_data.csv',index=False)
#files.download("20fold_rbb_1_1e4_wd_0_32_40_epoch4_clean_data.csv")