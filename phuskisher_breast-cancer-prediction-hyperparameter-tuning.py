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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
df=pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

df.head()
print(f'shape of dataset :{df.shape}')
df.info()

df.describe()
df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'])
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df, hue='diagnosis')
y=df['diagnosis']

X=df.drop('diagnosis',axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



print(f'X_train shape: {X_train.shape}')

print(f'X_test shape: {X_test.shape}')

print(f'y_train shape: {y_train.shape}')

print(f'y_test shape: {y_test.shape}')

lor=LogisticRegression(C=10)

lor.fit(X_train,y_train)

lor_pred=lor.predict(X_test)

lor_score=lor.score(X_test,y_test)
print(f'Score for Logistic Regression : {lor_score}')
cm_lor=confusion_matrix(y_test,lor_pred)

sns.heatmap(cm_lor, annot=True)

plt.title('Confusion Matrix: Logistic Regression')

plt.xlabel('predicted')

plt.ylabel('test')
param={'max_depth':np.linspace(1,20,20),

      'min_samples_split':[2,4,6,8],

      'min_samples_leaf':[1,2,3,4,5],

      'ccp_alpha':[0,0.01,0.1,1,10]}



dtc=DecisionTreeClassifier()

random_dtc=RandomizedSearchCV(estimator=dtc, param_distributions=param)

random_dtc.fit(X_train,y_train)

dtc_pred=random_dtc.predict(X_test)

dtc_score=random_dtc.score(X_test,y_test)
print(f'Score for Decision Tree Classification : {dtc_score}')
print(f' Optimised parameters for Decision tree classification :{random_dtc.best_params_}')
cm_dtc=confusion_matrix(y_test,dtc_pred)

sns.heatmap(cm_dtc, annot=True)

plt.title('Confusion Matrix: Decision Tree Classification')

plt.xlabel('predicted')

plt.ylabel('test')
param={'max_depth':np.linspace(1,20,20),

      'min_samples_split':[2,4,6,8],

      'min_samples_leaf':[1,2,3,4,5],

      'ccp_alpha':[0,0.01,0.1,1,10],

      'n_estimators':(np.linspace(100,1000,10)).astype(np.int32)}



rfc=RandomForestClassifier()

random_rfc=RandomizedSearchCV(estimator=rfc, param_distributions=param,n_jobs=-1)

random_rfc.fit(X_train,y_train)

rfc_pred=random_rfc.predict(X_test)

rfc_score=random_rfc.score(X_test,y_test)
print(f'Score for Random Forest Classification : {rfc_score}')
print(f' Optimised parameters for Random Forest Classification :{random_rfc.best_params_}')
cm_rfc=confusion_matrix(y_test,rfc_pred)

sns.heatmap(cm_rfc, annot=True)

plt.title('Confusion Matrix: Random Forest Classification')

plt.xlabel('predicted')

plt.ylabel('test')
gau=GaussianNB()



gau.fit(X_train,y_train)

gau_pred=gau.predict(X_test)

gau_score=gau.score(X_test,y_test)
print(f'Score for Naive Bayes Classification : {gau_score}')
cm_gau=confusion_matrix(y_test,gau_pred)

sns.heatmap(cm_gau, annot=True)

plt.title('Confusion Matrix: Naive Bayes Classification')

plt.xlabel('predicted')

plt.ylabel('test')
param={'C':[0.001, 0.01, 0.1, 1, 10],

    'gamma': [0.001, 0.01, 0.1, 1]}



svc=SVC()

random_svc=RandomizedSearchCV(estimator=svc, param_distributions=param,n_jobs=-1)

random_svc.fit(X_train,y_train)

svc_pred=random_svc.predict(X_test)

svc_score=random_svc.score(X_test,y_test)
print(f'Score for Support Vector Classification : {svc_score}')
print(f' Optimised parameters for Support Vector Classification :{random_svc.best_params_}')
cm_svc=confusion_matrix(y_test,svc_pred)

sns.heatmap(cm_svc, annot=True)

plt.title('Confusion Matrix: Support Vector Classification')

plt.xlabel('predicted')

plt.ylabel('test')