import csv as csv
import numpy as np
import pandas as pd
import random as rd

train_data=pd.read_csv('../input/train.csv')
#train_data.info()

#labels=np.array(train_data.columns)[[0,1,2,4,5,6,7,9,11]]
#labels=np.array(train_data.columns)[[0,1,2,4,6,7,11]]
labels=np.array(train_data.columns)[[0,1,2,4,6]]
#labels=np.array(train_data.columns)[[0,1,2,4,6]]
used_data=train_data.loc[:,labels]
used_data.info()
used_data.fillna('Nan',inplace=True)