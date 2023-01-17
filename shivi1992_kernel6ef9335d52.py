# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_digit = pd.read_csv('../input/train.csv')
test_data   = pd.read_csv('../input/test.csv')

import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
# Any results you write to the current directory are saved as output.
%matplotlib inline
(train_digit['label'].value_counts()/train_digit.shape[0]*100).plot.pie()
train_digit_dummies = pd.get_dummies(train_digit)
print(train_digit_dummies.shape)

train, validate = train_test_split(train_digit_dummies, test_size = 0.3, random_state = 100)
print(train.shape,validate.shape)

train_x    = train.drop('label', axis = 1)
train_y    = train['label']

validate_x = validate.drop('label', axis = 1)
validate_y = validate['label']

print(train_x.shape,train_y.shape,validate_x.shape,validate_y.shape)
model_rf = RandomForestClassifier(random_state=100, n_estimators= 60,oob_score = True,max_features='sqrt')
model_rf.fit(train_x, train_y)
pred_result_rf = model_rf.predict(validate_x)
df_rf = pd.DataFrame({'actual': validate_y,'prediction': pred_result_rf})
print(accuracy_score(validate_y,pred_result_rf))
test_pred    = model_rf.predict(test_data)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId']=test_data.index + 1
df_test_pred.head()
df_test_pred[['ImageId', 'Label']].to_csv('submission_1.csv', index = False)
df_test_pred.head()