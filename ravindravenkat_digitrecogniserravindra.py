# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
train.shape[0]
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
first_digit = train.iloc[3].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
train,validate = train_test_split(train,test_size=0.3,random_state=100)
train_x=train.drop('label',axis=1)
validate_x=validate.drop('label',axis=1)
train_y=train['label']
validate_y=validate['label']
#Decision Tree Classifier
model1 = DecisionTreeClassifier(random_state=100)
model1.fit(train_x,train_y)
test_pred1 = model1.predict(validate_x)
df_pred1 = pd.DataFrame({'actual':validate_y,'predicted':test_pred1})
df_pred1['pred_status'] = df_pred1['actual'] == df_pred1['predicted']
df_pred1['pred_status'].sum()/df_pred1.shape[0] *100
#Random Forest Classifer
model2 = RandomForestClassifier(random_state=100)
model2.fit(train_x,train_y)
test_pred2 = model2.predict(validate_x)
df_pred2 = pd.DataFrame({'actual':validate_y,'predicted':test_pred2})
df_pred2['pred_status'] = df_pred2['actual'] == df_pred2['predicted']
df_pred2['pred_status'].sum()/df_pred2.shape[0] *100
#Ada Boost Classifier
model3 = AdaBoostClassifier(random_state=100)
model3.fit(train_x,train_y)
test_pred3 = model3.predict(validate_x)
df_pred3 = pd.DataFrame({'actual':validate_y,'predicted':test_pred3})
df_pred3['pred_status'] = df_pred3['actual'] == df_pred3['predicted']
df_pred3['pred_status'].sum()/df_pred3.shape[0] *100
#knn algorithm 
from sklearn.neighbors import KNeighborsClassifier  
model4 = KNeighborsClassifier()
model4.fit(train_x,train_y)
test_pred4 = model4.predict(validate_x)
df_pred4 = pd.DataFrame({'actual':validate_y,'predicted':test_pred4})
df_pred4['pred_status'] = df_pred4['actual'] == df_pred4['predicted']
df_pred4['pred_status'].sum()/df_pred4.shape[0] *100
#knn algorithm for whole test data
train = pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train_y=pd.DataFrame()
train_x=pd.DataFrame()
train_y['label'] =train['label']
train_x=train.drop('label',axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x_scaled = pd.DataFrame(scaler.transform(train_x),columns=train_x.columns)
test_x_scaled=pd.DataFrame(scaler.transform(test),columns=test.columns)

from sklearn.neighbors import KNeighborsClassifier
model  =KNeighborsClassifier(n_neighbors=5)
model.fit(train_x_scaled,train_y)
test_pred =model.predict(test_x_scaled)

submission = pd.DataFrame(test_pred, columns=['Label'])
submission['ImageId'] = test.index + 1
submission[['ImageId', 'Label']].to_csv('submission.csv', index=False)