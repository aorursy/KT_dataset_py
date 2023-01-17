# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.neighbors import KNeighborsClassifier
import os
print(os.listdir("../input"))
train1 = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


train_x = train1.drop('label', axis = 1)
train_y = train1['label']

model_KNN = KNeighborsClassifier(n_neighbors=3)
model_KNN.fit(train_x, train_y)

test_pred = model_KNN.predict(test)
final_df = pd.DataFrame({'ImageId': list(range(1,len(test_pred)+1)), 'Label': test_pred})
final_df.to_csv('final_df.csv', index = False)

# Any results you write to the current directory are saved as output.
#from sklearn.ensemble import RandomForestClassifier

#model_rf = RandomForestClassifier(random_state=100)
#model_rf.fit(train_x, train_y)

#test_pred = model_rf.predict(test_x)
#df_pred = pd.DataFrame({'actual': test_y, 'predicted': test_pred})

#df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
#df_pred['pred_status'].sum()/df_pred.shape[0]*100
#from sklearn.tree import DecisionTreeClassifier

#model_dt = DecisionTreeClassifier(random_state=100)
#model_dt.fit(train_x, train_y)

#test_pred = model_dt.predict(test_x)
#df_pred = pd.DataFrame({'actual': test_y, 'predicted': test_pred})

#df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
#df_pred['pred_status'].sum()/df_pred.shape[0]*100
#from sklearn.ensemble import AdaBoostClassifier

#model_ab = AdaBoostClassifier(random_state=100)
#model_ab.fit(train_x, train_y)

#test_pred = model_ab.predict(test_x)
#df_pred = pd.DataFrame({'actual': test_y, 'predicted': test_pred})

#df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
#df_pred['pred_status'].sum()/df_pred.shape[0]*100