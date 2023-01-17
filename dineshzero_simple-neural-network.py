# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

merchants=pd.read_csv('../input/merchants.csv')

data_dict=pd.read_excel('../input/Data_Dictionary.xlsx')

hist_trans=pd.read_csv('../input/historical_transactions.csv')
hist_trans.head(10)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
train.columns
#model=RandomForestRegressor(n_estimators=1000)

#model.fit(train[['feature_1', 'feature_2', 'feature_3']],train['target'])
model=MLPRegressor(hidden_layer_sizes=(100,100,100))

model.fit(train[['feature_1', 'feature_2', 'feature_3']],train['target'])
test=pd.read_csv('../input/test.csv')
predicted=model.predict(test[['feature_1', 'feature_2', 'feature_3']])
predicted
output=pd.DataFrame()

output['card_id']=test['card_id']

output['target']=predicted

output.to_csv('output.csv',index=False)