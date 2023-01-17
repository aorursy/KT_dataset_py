# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train_y=train['label'] #output column
train_x=train.drop('label',axis=1)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)
pred=model.predict(test)
final = pd.DataFrame({'ImageId': list(range(1, len(pred)+1)), 'Label': pred})
final.to_csv('final.csv', index=False)
# p=pd.DataFrame(model_rf.predict(test))
#p.head()
#Any results you write to the current directory are saved as output.
# #decision tree
# hr=pd.read_csv('train.csv')
# train,test=train_test_split(hr,test_size=0.3,random_state=100)
# train_y=train['label'] #output column
# test_y=test['label']
# train_x=train.drop('label',axis=1)# i/p - all columns other than attrition(rowwise operation)
# test_x=test.drop('label',axis=1)
# model=DecisionTreeClassifier(random_state=100)#maxdepth is used to control the depth in the tree(only one division is done)
# model.fit(train_x,train_y)
# test_pre=model.predict(test_x)
# df_predictions = pd.DataFrame({'actual':test_y,'predicted':test_pre})
# df_predictions['pred_status']=df_predictions['actual']==df_predictions['predicted']
# df_predictions['pred_status'].value_counts()
# df_predictions['pred_status'].sum()/df_predictions.shape[0]*100
##ada boost

# hr=pd.read_csv('train.csv')
# from sklearn.ensemble import AdaBoostClassifier
# train,test=train_test_split(hr,test_size=0.3,random_state=100)
# train_y=train['label'] #output column
# test_y=test['label']
# train_x=train.drop('label',axis=1)# i/p - all columns other than attrition(rowwise operation)
# test_x=test.drop('label',axis=1)
# model=AdaBoostClassifier(random_state=100)
# model.fit(train_x,train_y)
# test_pred = model.predict(test_x)
# df_pred=pd.DataFrame({'actual':test_y,'predicted':test_pred})
# df_pred['pred_status'] = df_pred['actual']==df_pred['predicted']
# df_pred['pred_status'].sum()/df_pred.shape[0]*100
## random forest

# hr=pd.read_csv('train.csv')
# train,test=train_test_split(hr,test_size=0.3,random_state=100)
# train_y=train['label'] #output column
# test_y=test['label']
# train_x=train.drop('label',axis=1)# i/p - all columns other than attrition(rowwise operation)
# test_x=test.drop('label',axis=1)
# from sklearn.ensemble import RandomForestClassifier
# model_rf=RandomForestClassifier(random_state=100)
# model_rf.fit(train_x,train_y)

# test_pred = model_rf.predict(test_x)
# df_pred=pd.DataFrame({'actual':test_y,'predicted':test_pred})
# df_pred['pred_status'] = df_pred['actual']==df_pred['predicted']
# df_pred['pred_status'].sum()/df_pred.shape[0]*100
