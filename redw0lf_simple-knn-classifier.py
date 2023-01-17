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
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/train.csv')
label = data['label']
train_data = data.drop(['label','id'],axis=1)
data_id = data['id']
train_data = train_data.fillna(method='ffill')
train_data['feature_3'] =train_data['feature_3'].replace({'a':0 ,'b':1, 'c':'2','d':3, 'e':4})
train_data.head()
test_data = pd.read_csv('../input/test.csv')
test_data = test_data.fillna(method='ffill')
test_id = test_data['id']
test_data = test_data.drop(['id'],axis=1)
def get_replace_dict(categorical_feature):
    unique_row = categorical_feature.unique()
    replace_dict= dict()
    for i,letter in enumerate(unique_row):
        replace_dict[letter] = i
    return replace_dict
train_data['feature_10']= train_data['feature_10'].replace(get_replace_dict(train_data['feature_10']))
train_data['feature_3']= train_data['feature_3'].replace(get_replace_dict(train_data['feature_3']))
test_data['feature_10']= test_data['feature_10'].replace(get_replace_dict(test_data['feature_10']))
test_data['feature_3']= test_data['feature_3'].replace(get_replace_dict(test_data['feature_3']))
plt.scatter(train_data['feature_0'],train_data['feature_4'],c=label)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data,label)
print('Accuracy: {}'.format(neigh.score(train_data,label)))
print('F1_score: {}'.format(f1_score(label,neigh.predict(train_data),average='macro')))
pred_label = neigh.predict(test_data)
predicted_data = pd.DataFrame([test_id,pred_label]).transpose()
predicted_data.columns = ['id ','label']
predicted_data.to_csv('submission.csv',index=False,header=True)

