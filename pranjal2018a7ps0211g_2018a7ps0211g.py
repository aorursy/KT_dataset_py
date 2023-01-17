# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/minor-project-2020/train.csv")

df.shape

df.head()
df.info()
# pd.set_option('display.max_columns', None)

# pd.set_option('max_columns', None)

# pd.set_option("max_rows", None)

df.corr()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df.drop(['id'],axis=1,inplace=True)
df.head()

df.shape
X=df.drop(['target'],axis=1)

Y=df['target']

print(Y)
# import xgboost

# classifier=xgboost.XGBRegressor()

# classifier.fit(X,Y)
# Ytest=classifier.predict_proba(Xtest)
# print(Ytest)
# dat=pd.DataFrame(temp)
dft=pd.read_csv("../input/minor-project-2020/test.csv")

idtest=dft['id']

Xtest=dft.drop(['id'],axis=1)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X)

scaled_X_test = scalar.transform(Xtest)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt = DecisionTreeClassifier()
dt.fit(scaled_X_train,Y)
y_pred = dt.predict(scaled_X_test)
dat=pd.DataFrame(y_pred)
ans=pd.read_csv("../input/minor-project-2020/sample_submission.csv")

datasets=pd.concat([idtest,dat],axis=1)

datasets.columns=['id','target']

datasets.to_csv('samplesubmission.csv',index=False)
# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=1)



# clf.fit(scaled_X_train,Y)
# y_pred = clf.predict(scaled_X_test)
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(scaled_X_train,Y)
y_pred = xgb.predict(scaled_X_test)
dat=pd.DataFrame(y_pred)
ans=pd.read_csv("../input/minor-project-2020/sample_submission.csv")

datasets=pd.concat([idtest,dat],axis=1)

datasets.columns=['id','target']

datasets.to_csv('samplsubmission.csv',index=False)