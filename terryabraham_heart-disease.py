# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.isnull().sum()
X = df.drop('target', axis=1)

X = StandardScaler().fit_transform(X)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model=SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)

roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))
model1= RandomForestClassifier(n_estimators=1000)

model1.fit(X_train, y_train)

predictions = cross_val_predict(model1, X_test, y_test, cv=5)

print(classification_report(y_test, predictions))
score1= np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))

np.around(score1, decimals=4)
model2=KNeighborsClassifier()

model2.fit(X_train,y_train)

predictions=cross_val_predict(model2,X_test,y_test,cv=5)

print(classification_report(y_test, predictions))
score2= np.around(np.mean(cross_val_score(model2, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score2))
model3=LogisticRegression()

parameters={'C':[0.001,0.01,0.1,1,10,100]}

grid = GridSearchCV(estimator=model3, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
score3= np.around(np.mean(cross_val_score(model3, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score3))
names=[]

scores=[]

names.extend(['SVC','RF','KNN','LR'])

scores.extend([roc_auc,score1,score2,score3])

algorithms=pd.DataFrame({'Score':scores},index=names)

print("Most accurate : \n{}".format(algorithms.loc[algorithms['Score'].idxmax()]))