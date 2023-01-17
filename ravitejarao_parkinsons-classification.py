# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv('/kaggle/input/parkinsons-data-set/parkinsons.data')
pd.set_option('display.max_columns', None)
df
df.describe()
df['status'].value_counts(normalize=True)*100
df.info()
plt.figure(figsize=(15,8))
sns.pairplot(data=df,hue='status')
plt.show()
plt.figure(figsize=(15,8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)
plt.show()
X=df[[ 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE']]
y=df['status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import randint

est = RandomForestClassifier(n_jobs=-1)
rf_p_dist={'max_depth':[1,2,3,4,5,6,7,8,9,10],
           'n_estimators':[1,2,3,4,5,10,11,12,13,100,200,300,400,500],
              'max_features':[1,2,3,4,5],
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':[1,2,3,4,5]
         
              
              }
         
rs=GridSearchCV(estimator=est,param_grid=rf_p_dist)
rs.fit(X_train,y_train)
rs.estimator
rf=RandomForestClassifier(criterion='entropy', max_depth=3, max_features=2,
                       min_samples_leaf=3,random_state=0)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(classification_report(y_test,y_pred))
print('train')
print(rf.score(X_train,y_train))
print('test')
print(rf.score(X_test,y_test))