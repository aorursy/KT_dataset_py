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
feature_importance = pd.read_csv('../input/lightgbm-basic-data/feature_importance.csv')

del feature_importance['fold']

feature_importance = feature_importance.groupby(by=feature_importance['feature']).sum()

feature_importance = feature_importance.sort_values(by='importance',ascending=False)

feature_importance.head()
feature_importance['importance'].plot()
train_data = pd.read_csv('../input/data-preprocessing/treemodel_train.csv')

test_data = pd.read_csv('../input/data-preprocessing/treemodel_test.csv')

train_data.head()
label = train_data['label']

del train_data['label']

data = train_data.append(test_data)

data.head()
data['try_distant_day_mean'] = np.array(data['distant_day_mean']).astype(int)

data = pd.get_dummies(data,columns=['try_distant_day_mean'])

data.head()
data['principal_cut'] = pd.cut(data['principal'], 10, labels=False)
for i in range(32):

    column = 'try_distant_day_mean_'+str(i-1)

    try_df = data[['principal_cut',column]].groupby(data['principal_cut']).mean()

    try_df.columns = ['principal_cut','principal_distant_day'+str(i)]

    data = pd.merge(data,try_df,on='principal_cut',how='left')

    del data[column]

data.head()

    
data['creat_sub'] = data['one_distant_day_max'] - data['one_distant_day_mean']

data['creat_sub2'] = data['distant_day_mean'] - data['one_distant_day_mean']

data['creat_add'] = data['one_distant_day_max'] + data['one_distant_day_mean']

data['creat_add2'] = data['one_distant_day_mean'] +data['distant_day_mean']

data['creat_multip'] = data['principal']/data['all_repay_amt_sum']
train_data = data[:len(train_data)]

test_data = data[len(train_data):]

train_data['label'] = label
train_data.to_csv('treemodel_train.csv',index=False)

test_data.to_csv('treemodel_test.csv',index=False)