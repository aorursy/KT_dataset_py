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
!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!pip install pylg
!pip install GPUtil

# issues in certain dependency packages like pillow, these two requirements seem to be enough for kaggle. already pillow 5.4.1 available here
import os
import sys
sys.path.append('monk_v1/monk/');

from pytorch_prototype import prototype

ptf = prototype(verbose=1)
ptf.Prototype('oregon-wildlife', 'oregon-pytorch')
'''
**here**
use md5 hashing to delete duplicates 
'''

data_dir = '../input/oregon-wildlife/oregon_wildlife/oregon_wildlife/'

ptf.Dataset_Params(dataset_path = data_dir, split = 0.8)

ptf.apply_random_horizontal_flip(train=True, val=True)
ptf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True) 
# dont know what how to choose values for preprocessing. copy pasted this.

ptf.Dataset()
ptf.Model_Params(model_name = 'vgg16')
ptf.Model()
ptf.Freeze_Layers(num = 13)
# training the final conv layer and the 3 fcn layers of the architecture. should give good results i think.
lrs = [0.07, 0.01, 0.03]  # usually 1e-2 is a good learning rate so keeping lrs elements around that
percent_data = 5
ptf.optimizer_sgd(0.01, weight_decay = 0.01)
ptf.update_batch_size(4)
ptf.loss_softmax_crossentropy()
ptf.Reload()
analysis = ptf.Analyse_Learning_Rates('lr_cycle', lrs, percent_data, 
                                      num_epochs=5) # dont know what state argument does

""" 
dont know how to resolve this bug. I have tried this several times before but never faced this issue. need help here.
"""
analysis = ptf.Analyse_Optimizers('optim_cycle', optimizers, percent_data, 
                                      num_epochs=epochs, state="keep_none")
analysis = ptf.Analyse_Batch_Sizes('batch-cycle', batch_sizes, percent_data,
                                  num_epochs = epochs, state='keep_none')
#selecting best hyperparams

ptf.update_batch_size(4)
ptf.update_learning_rate(0.01)
ptf.optimizer_sgd(0.01, weight_decay = 0.01)

ptf.Reload()
ptf.Dataset()
# after reloading number of trainable layers changes, I think it's a bug it shouldn't change 

ptf.Freeze_Layers(num=13)
ptf.Training_Params(num_epochs = 10,
                display_progress = True,
                display_progress_realtime = True)

ptf.loss_softmax_crossentropy()

ptf.Train()
'''
dont know how to resolve this. need help here.
'''
# batch_sizes = [4, 8, 16]  # keeping batch sizes of 2^n helps in computation
# optimizers = ['sgd', 'adam', 'momentum_rmsprop']
# percent_data = 5  # around 500 images
# epochs = 5