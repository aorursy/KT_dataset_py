# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('/kaggle/input/spotifyclassification/data.csv')



df = df.drop(columns=['Unnamed: 0','song_title','artist'])

print(df.head())

col_here = df.columns.tolist()

print(col_here)
fig,axs = plt.subplots(4,4,figsize=(20, 15))

x = 0

for i in range(0,4):

    for j in range(0,4): 

        try:

            p = df[df['target']==1][col_here[x]]

            n = df[df['target']==0][col_here[x]]

            sns.distplot(p,color='g',ax=axs[i,j],bins=40)

            sns.distplot(n,color='r',ax=axs[i,j],bins=40)

            x += 1

            

        except:

            pass

    

    

    
# print(df.corr())



sns.heatmap(df.corr())

# sns.heatmap(df[df['target']==1].corr())

# sns.heatmap(df[df['target']==0].corr())
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = df.drop(columns=['target'])

y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
dt_clf = DecisionTreeClassifier(max_depth=6,random_state=4)

dt_clf.fit(X_train,y_train)

dt_score = dt_clf.score(X_test,y_test)

print('Decision tree score is ',dt_score)



ada_clf = AdaBoostClassifier(base_estimator=dt_clf,random_state=0)

ada_clf.fit(X_train,y_train)

ada_score = ada_clf.score(X_test,y_test)

print('AdaBoost with Decision tree estimator score is ',ada_score)



gb_clf = GradientBoostingClassifier(random_state=0)

gb_clf.fit(X_train,y_train)

gb_score = gb_clf.score(X_test,y_test)

print('Gradiant Boosting Score is ',gb_score)
from xgboost import XGBClassifier



xgb_clf = XGBClassifier(base_estimator=gb_clf,random_state=5)

xgb_clf.fit(X_train,y_train)

xgb_score = xgb_clf.score(X_test,y_test)



print("XGBoost score is ",xgb_score)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3,0.9],

            'max_depth':range(1,20)}



clf_model = GridSearchCV(estimator=xgb_clf,param_grid=parameters)

clf_model.fit(X_train,y_train)

y_pred = clf_model.predict(X_test)



clf_score = accuracy_score(y_test,y_pred)

clf_cm = confusion_matrix(y_test,y_pred)

clf_cr = classification_report(y_test,y_pred)



print('XGB score is ',xgb_score)

print('XGB with grid search score is ',clf_score)