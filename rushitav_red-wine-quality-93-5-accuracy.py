# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib notebook

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

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
data=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data
data.shape
data.describe()
data.isnull().sum()
data.quality.unique()
data.quality.value_counts()
data['quality']=data.quality.apply(lambda x:1 if x>6.5 else 0)

data
data.quality.value_counts()

plt.figure()

plt.hist(data['quality'])

plt.show()
sns.pairplot(data,hue='quality',diag_kind='hist')

plt.show()
sns.countplot(data=data,x='quality')
fig=plt.figure(figsize=[20,15])

features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

count=1

for i in features:

    plt.subplot(3,4,count)

    sns.barplot(data['quality'],data[i],hue=data['quality'])

    count=count+1

plt.show()
fig=plt.figure(figsize=[10,10])

sns.heatmap(data.corr(),annot=True,center=0,cmap='Blues')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X=data[features]

y=data['quality']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

model=LogisticRegression()

model.fit(X_train_scaled,y_train)

predict=model.predict(X_test_scaled)

score=accuracy_score(predict,y_test)

score
from sklearn.svm import SVC

model=SVC()

model.fit(X_train_scaled,y_train)

predict=model.predict(X_test_scaled)

score=accuracy_score(predict,y_test)

score
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(random_state=0)

model.fit(X_train_scaled,y_train)

predict=model.predict(X_test_scaled)

score=accuracy_score(predict,y_test)

score
from sklearn.model_selection import GridSearchCV

param={'n_estimators':[5,10,15,20,50,100,200,500]}



model=RandomForestClassifier(random_state=0)

grid_acc=GridSearchCV(model,param_grid=param)

grid_acc.fit(X_train_scaled,y_train)

 



print('Grid best parameter:',grid_acc.best_params_)

print('Grid best score for Random Forest:',grid_acc.best_score_)
model_2=RandomForestClassifier(n_estimators=20,random_state=0)

model_2.fit(X_train_scaled,y_train)

predict=model_2.predict(X_test_scaled)

score=accuracy_score(predict,y_test)

score