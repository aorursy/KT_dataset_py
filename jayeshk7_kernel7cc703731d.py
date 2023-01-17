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
import os
import sys
sys.path.append('monk_v1/monk/')
from pytorch_prototype import prototype

ptf = prototype(verbose=1)
ptf.Prototype('oregon-wildlife', 'oregon-pytorch')
data_dir = '../input/oregon-wildlife/oregon_wildlife/oregon_wildlife/'

ptf.Default(dataset_path = data_dir,
           model_name = 'vgg16',
           freeze_base_network = True,
           num_epochs = 8)

ptf.update_trainval_split(0.8)
ptf.Reload() 

ptf.update_batch_size(4)
ptf.Reload()

ptf.Freeze_Layers(num=13)
lrs = [0.01, 0.03, 0.06]
percent_data=5
epochs=5

analysis1 = ptf.Analyse_Learning_Rates('lr-cycle', lrs, percent_data, num_epochs=epochs, state='keep_none')
optimizers = ['sgd', 'adam', 'momentum_rmsprop']

analysis2 = ptf.Analyse_Optimizers('optim-cycle', optimizers, percent_data, num_epochs=epochs, state='keep_none')
batch_sizes = [4,8,16]

analysis3 = ptf.Analyse_Batch_Sizes('batch-cycle', batch_sizes, percent_data, num_epochs=5, state='keep_none')
ptf.update_batch_size(8)
ptf.Reload()
ptf.optimizer_sgd(0.01, weight_decay=0.01)
ptf.Reload()
ptf.Freeze_Layers(num=13)
ptf.Train()
ptf.prototype(verbose=1)
ptf.Prototype('oregon-wildlife', 'oregon-pytorch-unfreezed')

ptf.Default(dataset_path = data_dir,
           model_name = 'vgg16',
           freeze_base_network = False,
           num_epochs = 8)

ptf.update_trainval_split(0.8)
ptf.Reload()
ptf.Analyse_Optimizers('optim')