# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
# Any results you write to the current directory are saved as output.
#data_dummies = pd.get_dummies(data)
#train, validate = train_test_split(data_dummies, test_size = 0.2, random_state = 100)

train_y = data['label'] 
#validate_y = validate['label'] 

train_x = data.drop('label', axis = 1) 
#validate_x = validate.drop('label', axis = 1)
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train_x, train_y)

test_pred_knn = model_knn.predict(test)
df_pred_knn = pd.DataFrame({'ImageId': list(range(1, len(test_pred_knn)+1)), 'Label': test_pred_knn})
df_pred_knn.to_csv('submission.csv', index = False)
#model_decision = DecisionTreeClassifier(random_state=100)
#model_decision.fit(train_x, train_y)
"""test_pred_decision = model_decision.predict(validate_x)
df_pred_decision = pd.DataFrame({'actual': validate_y,
                       'Predicted': test_pred_decision})
df_pred_decision['Status'] = df_pred_decision['actual'] == df_pred_decision['Predicted']
df_pred_decision['Status'].sum()/df_pred_decision.shape[0] * 100"""
"""model_rf = RandomForestClassifier(random_state=100, n_estimators=100)
model_rf.fit(train_x, train_y)"""
"""test_pred_rf = model_rf.predict(validate_x)
df_pred_rf = pd.DataFrame({'actual': validate_y,
                       'Predicted': test_pred_rf})
df_pred_rf['Status'] = df_pred_rf['actual'] == df_pred_rf['Predicted']
df_pred_rf['Status'].sum()/df_pred_rf.shape[0] * 100"""
"""model_ada = AdaBoostClassifier(random_state=100)
model_ada.fit(train_x, train_y)"""
"""test_pred_ada = model_ada.predict(validate_x)
df_pred_ada = pd.DataFrame({'actual': validate_y,
                       'Predicted': test_pred_ada})
df_pred_ada['Status'] = df_pred_ada['actual'] == df_pred_ada['Predicted']
df_pred_ada['Status'].sum()/df_pred_ada.shape[0] * 100"""