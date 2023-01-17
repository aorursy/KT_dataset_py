# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.columns
df.info()
df.isnull().sum()
df.hist(bins=20)
sns.countplot(x='Outcome',data=df)
plt.figure(figsize=(10,15))
p=1
for i in df.columns:
    plt.subplot(3,3,p)
    sns.barplot(x='Outcome',y=i,data=df)
    p=p+1
plt.figure(figsize=(10,15))
p=1
for i in df.columns:
    plt.subplot(3,3,p)
    sns.violinplot(x='Outcome',y=i,data=df)
    p=p+1
sns.pairplot(data=df,hue='Outcome',diag_kind='kde')
plt.show()
df.head()
# Feature scaling with StandardScaler
from sklearn.preprocessing import StandardScaler
scale_features_std = StandardScaler()
X = scale_features_std.fit_transform(df.drop(['Outcome'],axis=1))
Y = df[['Outcome']]
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25)
abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
for i in models:
    model = i
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    abc.append(metrics.accuracy_score(prediction,test_Y))
models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe.columns=['Accuracy']
models_dataframe
sns.heatmap(df.drop(['Outcome'],axis=1).corr(),annot=True,cmap='RdYlGn')
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
svc = SVC()
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(train_X,train_Y)
grid_svc.best_params_
model = svm.SVC(kernel='linear',C=.8,gamma=.1)
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print(metrics.accuracy_score(prediction,test_Y))
