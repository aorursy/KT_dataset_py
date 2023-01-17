# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Download all model Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from numpy import loadtxt
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']
rf = RandomForestRegressor(n_estimators= 472,
 min_samples_split= 2,
 min_samples_leaf= 2,
 max_features= 'auto',
 max_depth= 300,
 bootstrap= True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=101)
m3=SVC()
m3.fit(X_train,y_train)
m3.score(X_test,y_test)
m4=DecisionTreeClassifier()
m4.fit(X_train, y_train)
m4.score(X_test,y_test)
rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'F3':'F17']
pred = rf.predict(df_test)
print(pred)

result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output707.csv', index=False)

