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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.sample(5)
df_train.info()
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(14,14))
sns.heatmap(df_train.isnull())
fig = plt.figure(figsize=(10,8))
sns.heatmap(df_train.corr(),annot=True)
df_train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df_train
avg_age = df_train['Age'].mean()
avg_age
df_train['Age'].fillna(avg_age,inplace=True)
sex_train = pd.get_dummies(df_train['Sex'],drop_first=True)
sex_train.head()
df_train = pd.concat([df_train,sex_train],axis=1)
df_train.drop('Sex',axis=1,inplace=True)
df_train.info()
X = df_train.drop('Survived',axis=1)
y = df_train['Survived']
print(X.shape)
print(y.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)
classifiers = [
    DecisionTreeClassifier(),
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(10, 10, 10),max_iter=1000)
]
models = []
print("Train Test Split\n")
for cls in classifiers:
    name = cls.__class__.__name__
    mod = cls.fit(X_train, y_train)
    predicted = mod.predict(X_test)
    
    print(name)
    print('*'*30)
    print('Accuracy  : ',accuracy_score(y_test,predicted))
    print('Recall    : ',recall_score(y_test,predicted))
    print('Precision : ',precision_score(y_test,predicted))
    print('F1 Score  : ',f1_score(y_test,predicted))
    print("")
    models.append(mod)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
class ClassificationModel:
   
    def __init__(self,name, accuracy, recall,precision,f1):
        self.name = name
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
    
   
    def display(self):
        print(self.name)
        print('*'*30)
        print('Accuracy     : ',self.accuracy)
        print('Recall       : ',self.recall)
        print('Precision    : ',self.precision)
        print('F1 Score     : ',self.f1)
        print('Avg F1 Score :',np.array(self.f1).mean())
        print('')
        
print('K fold cross validation\n')

model_object = []
for cls in classifiers:
    name = cls.__class__.__name__
    accuracy = []
    recall = []
    precision = []
    f1 = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
    
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        cls.fit(X_train, y_train)
        predicted = cls.predict(X_test)

        accuracy.append(accuracy_score(y_test, predicted))
        recall.append(recall_score(y_test, predicted))
        precision.append(precision_score(y_test, predicted))
        f1.append(f1_score(y_test, predicted))
        
    model_object.append(ClassificationModel(name,accuracy,recall,precision,f1))

for obj in model_object:
    obj.display()

