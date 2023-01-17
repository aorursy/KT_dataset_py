# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head(5)

data.shape
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
test
data.describe()
data.info()
data.isnull().sum()
data.isnull().sum()/data.shape[0] *100
import seaborn as sns
sns.heatmap(data.isnull())
data=data.drop('Cabin',axis=1)
data.head(5)
test=test.drop('Cabin',axis=1)
test.head(5)
data.shape
data['Survived'].value_counts()
sns.countplot(data['Survived'])
cols=['Sex','Pclass','SibSp','Parch','Fare','Embarked']
n_rows=2
n_cols=3
l=0
fig,axs=plt.subplots(2,3, figsize=(16,9))
for i in range(n_rows):
    for j in range(n_cols):
        l=l+1
        ax=axs[i][j]
        ax.set_title(cols[i*n_cols +j])
        sns.countplot(data[cols[i*n_cols+j]],hue=data['Survived'],ax=ax)
data.groupby('Sex')[['Survived']].mean()*100
data.groupby('Pclass')[['Survived']].mean()
data.pivot_table('Survived',index='Sex',columns='Pclass')
data.pivot_table('Survived',index='Sex',columns='Pclass').plot()
sns.barplot(x='Pclass',y='Survived',data=data)
age=pd.cut(data['Age'],[0,18,80])
data.pivot_table('Survived',['Sex',age],'Pclass')
plt.figure(figsize=(16,9))
sns.scatterplot(x=data['Fare'],y=data['Pclass'],hue=data['Survived'],alpha=0.9)
data=data.drop('Name',axis=1)
test=test.drop('Name',axis=1)
data.dtypes
print(data['Sex'].unique())
print(data['Embarked'].unique())
# from sklearn.impute import SimpleImputer
# impute=SimpleImputer(strategy='mean')
# impute.fit(data['Age'])
data['Age']=data['Age'].fillna(data['Age'].mean())
data['Age'].isnull().sum().sum()
test['Age']=data['Age'].fillna(test['Age'].mean())
test['Age'].isnull().sum().sum()
# data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode(),)
# data['Embarked'].unique()
data=data.dropna(axis=0)
test=test.dropna(axis=0)
data.isnull().sum().sum()
test.dtypes
data.dtypes
sns.heatmap(data.isnull())
from sklearn.preprocessing import LabelEncoder
labelen=LabelEncoder()
data.iloc[:,3]=labelen.fit_transform(data.iloc[:,3].values)
labelen=LabelEncoder()
data.iloc[:,9]=labelen.fit_transform(data.iloc[:,9].values)
data['Sex']
labelen=LabelEncoder()
test.iloc[:,2]=labelen.fit_transform(test.iloc[:,2].values)
labelen=LabelEncoder()
test.iloc[:,8]=labelen.fit_transform(test.iloc[:,8].values)
data['Sex']
data['Embarked']
data.dtypes

x_train=data.drop(['Survived','Ticket'],axis=1)
y_train=data['Survived']
test=test.drop(['Ticket'],axis=1)
models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVC',SVC()))
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    cv_result=cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))
    print(msg)
from sklearn.preprocessing import StandardScaler
x_train=x_train.drop('PassengerId',axis=1)
sc_x=StandardScaler()
xsc=sc_x.fit_transform(x_train)
test1=test.drop('PassengerId',axis=1)
test1=sc_x.fit_transform(test1)
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    cv_result=cross_val_score(model,xsc,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))
    print(msg)
x_train,x_test,y_train,y_test=train_test_split(xsc,y_train,test_size=0.2,random_state=1)
my_model=SVC()
my_model.fit(x_train,y_train)
y_pred=my_model.predict(x_test)
test1.shape
print('Training Accuracy:- ',my_model.score(x_train,y_train))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Confusion_matrix:\n',confusion_matrix(y_test,y_pred))
print('Accuracy:- ',accuracy_score(y_test,y_pred)*100,'%')
print('Classification Report:-\n',classification_report(y_test,y_pred))

predictions = my_model.predict(test1)
output = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': predictions})
output.to_csv('titanic.csv', index=False)
output
#print("Your submission was successfully saved!")
