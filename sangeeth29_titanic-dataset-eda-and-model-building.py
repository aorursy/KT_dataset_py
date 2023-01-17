import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
eda = pd.read_csv('../input/titanic-dataset/test.csv')
eda.head()
eda.info()
eda.describe(include='all')
sns.countplot(x = 'Sex', data = eda)
sns.countplot(x='Embarked',data=eda)
sns.boxplot(y='PassengerId',data=eda)
sns.boxplot(y='Age',data=eda)
sns.distplot(eda['Age'],bins=10)
sns.boxplot(y='SibSp',data=eda)
sns.distplot(eda['SibSp'])
sns.boxplot(y= 'Fare',data=eda)
sns.distplot(eda['Fare'],bins=5)
plt.figure(figsize=(15,6))

plt.scatter(eda['PassengerId'],eda['Age'])

plt.title('PassengerId VS Age')

plt.xlabel('PassengerId')

plt.ylabel('Age')
plt.figure(figsize=(15,6))

plt.scatter(eda['PassengerId'],eda['SibSp'])

plt.title('PassengerId VS SibSp')

plt.xlabel('PassengerId')

plt.ylabel('SibSp')
plt.figure(figsize=(15,8))

plt.scatter(eda['PassengerId'],eda['Fare'])

plt.title('PassengerId VS Fare')

plt.xlabel('PassengerId')

plt.ylabel('Fare')
plt.figure(figsize=(15,8))

plt.scatter(eda['Age'],eda['SibSp'])

plt.title('Age VS SibSp')

plt.xlabel('Age')

plt.ylabel('SibSp')
plt.figure(figsize=(15,8))

plt.scatter(eda['Age'],eda['Fare'])

plt.title('Age VS Fare')

plt.xlabel('Age')

plt.ylabel('Fare')
sns.heatmap(eda.corr(),annot=True,linewidth=0.5)
eda_cont=eda.iloc[:,:-3]

eda_cont

sns.pairplot(eda_cont)
A = eda.groupby(['Sex','Embarked'],axis = 0)

A.size()
pd.crosstab(eda['Sex'],eda['Embarked']).plot(kind='bar',stacked=True)
eda.info()
eda.describe()
eda.isnull().sum()
eda['Age'].describe()
eda.hist(column=['Age'],bins=5)
eda['Age']=eda['Age'].fillna(value=eda['Age'].median())
mode = eda['Cabin'].mode().values[0]

eda['Cabin'].fillna(value=mode,inplace=True)
mode1 = eda['Embarked'].mode().values[0]

eda['Embarked'].fillna(value=mode1,inplace=True)
eda
eda.info()
eda
eda.boxplot(column=['Age'])
eda.boxplot(column=['Fare'])
plt.scatter(eda['Age'],eda['Fare'])
eda['Age'].describe()
IQR=eda['Age'].quantile(0.75)-eda['Age'].quantile(0.25)

print(IQR)
Upper_Outlier_Limit = eda['Age'].quantile(0.75) + 1.5*IQR

Upper_Outlier_Limit
Lower_Outlier_Limit = eda['Age'].quantile(0.25) - 1.5*IQR

Lower_Outlier_Limit
Outlier_Values = eda[(eda['Age']>=Upper_Outlier_Limit)|(eda['Age']<=Lower_Outlier_Limit)]
Outlier_Values
Upper_Outlier_Limit1 = eda['Fare'].quantile(0.75) + 1.5*IQR

Upper_Outlier_Limit1
Lower_Outlier_Limit1 = eda['Fare'].quantile(0.25) - 1.5*IQR

Lower_Outlier_Limit1
Outlier_Values1 = eda[(eda['Fare']>=Upper_Outlier_Limit1)|(eda['Fare']<=Lower_Outlier_Limit1)]
Outlier_Values1
eda.drop('Cabin',axis=1,inplace=True)
eda.drop('Ticket',axis=1,inplace=True)
eda.drop('Pclass',axis =1,inplace =True)

eda.drop('Name',axis=1,inplace=True)

eda.drop('Sex',axis=1,inplace =True)

eda.drop('SibSp',axis=1,inplace=True)

eda.drop('Parch',axis=1,inplace=True)

eda.drop('Fare',axis=1,inplace =True)

eda.drop('Embarked',axis=1,inplace =True)
obj=eda.dtypes==np.object

print(obj)
obj=eda.dtypes==np.object

print(obj)
eda.columns[obj]
eda = pd.get_dummies(eda,drop_first=True)
eda
eda.head()
cols=eda.columns

cols=['PassengerId','Age']
eda = eda[cols]

eda
X = eda.iloc[:,:-1].values

Y = eda.iloc[:,-1].values
X
Y
X.shape
Y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape