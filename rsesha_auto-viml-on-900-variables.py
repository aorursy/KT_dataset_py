# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/ram-reduce/reduce_train.csv')
print(train.shape)
train.head()
!pip install autoviml
from autoviml.Auto_ViML import Auto_ViML
m, feats, trainm, testm = Auto_ViML(train, target='accuracy_group', test='',
                            sample_submission='',
                            scoring_parameter='', KMeans_Featurizer=False,
                            hyper_param='GS',feature_reduction=True,
                             Boosting_Flag="CatBoost",Binning_Flag=False,
                            Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False,
                            verbose=0)
m
feats
