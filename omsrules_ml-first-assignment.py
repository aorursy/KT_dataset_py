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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
main_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
df_features = main_data.iloc[:, 1:785]
df_label = main_data.iloc[:, 0]
train_x,validate_x,train_y,validate_y = train_test_split(df_features, df_label, 
                                                test_size = 0.2,
                                                random_state = 1212)
print(train_x.shape, validate_x.shape, train_y.shape, validate_y.shape)

train_x = train_x.astype('float32'); validate_x= validate_x.astype('float32')
model_rf = RandomForestClassifier(random_state = 400 , n_estimators=400,oob_score = True, n_jobs = -1,max_features = "sqrt")
model_rf.fit(train_x,train_y)

validate_pred_rf = model_rf.predict(validate_x)
accuracy_score(validate_y , validate_pred_rf)#.9679
test_pred  = model_rf.predict(test_data)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId']=test_data.index + 1
df_test_pred.head()
df_test_pred[['ImageId', 'Label']].to_csv('submission_1.csv', index = False)
df_test_pred.head()
