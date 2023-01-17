# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

df.head(10)
df.describe()
df.isnull().sum()
df['diagnosis'].value_counts()
sns.set(style='ticks')

sns.pairplot(df,hue='diagnosis')
df.corr()
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),cmap='Dark2',annot=True,linewidths=2.0,linecolor='black')
from sklearn.model_selection import train_test_split

X = df.drop('diagnosis',axis=1)

y = df['diagnosis']
from sklearn.preprocessing import StandardScaler

X_copy = X

X = StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)

y_test.shape,y_pred1.shape
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred1))
n_estimators=[100,200,500,750,1000,1100,1200]

max_depth=[3,5,10,15,20]

booster=['gbtree']

learning_rate=[0.03, 0.06, 0.1, 0.15, 0.2]

min_child_weight=[1,2,3,4]

base_score=[0.2,0.25, 0.5, 0.75]



hyperparameter_grid={'n_estimators':n_estimators,

                     'max_depth':max_depth,

                     'learning_rate':learning_rate,

                     'min_child_weight':min_child_weight,

                     'booster':booster,

                     'base_score':base_score}
import xgboost as xgb

classifier = xgb.XGBClassifier()
y1 = df[['diagnosis']]

X_train,X_test,y1_train,y1_test = train_test_split(X,y1,test_size=0.25)
from sklearn.model_selection import RandomizedSearchCV

randomcv = RandomizedSearchCV(estimator=classifier,

                              param_distributions = hyperparameter_grid,

                              n_iter=50,

                              verbose=5,

                              n_jobs=4,

                              return_train_score=True,

                              random_state=42)
randomcv.fit(X_train,y1_train)
randomcv.best_estimator_
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.03, max_delta_step=0, max_depth=3,

              min_child_weight=4, missing=None, monotone_constraints='()',

              n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
model.fit(X_train,y1_train)
y_pred2 = model.predict(X_test)
print(accuracy_score(y1_test,y_pred2))