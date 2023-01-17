!pip install dominance-analysis
import numpy as np 

import pandas as pd 

import os

from dominance_analysis import Dominance



train = pd.read_csv('../input/train.csv')

train.head()
cols = [c for c in train.columns if c not in ['id', 'target']]

cols2 = [c for c in train.columns if c not in ['id']]



# Negative values in columns are not allowed, so we will use MinMaxScaler.



from sklearn import preprocessing    

scaler = preprocessing.MinMaxScaler()

train_scaled = pd.DataFrame(scaler.fit_transform(train[cols]))
train_scaled['target']=train['target']

train_scaled.columns = train[cols2].columns

train_scaled.head()
# We will try with only 10 features, default is 15

dominance_classification=Dominance(data=train_scaled,target='target',objective=0,pseudo_r2="mcfadden",top_k=10)
dominance_classification.incremental_rsquare()
dominance_classification.plot_incremental_rsquare()
dominance_classification.dominance_stats()