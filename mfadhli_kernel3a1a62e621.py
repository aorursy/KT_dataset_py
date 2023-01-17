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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
train_data.columns
test_data.columns
train_data.dtypes
train_data.shape, test_data.shape
train_data['Survived'].value_counts()
train_data['Survived'].value_counts(normalize=True)
train_data['Survived'].value_counts(normalize=True).plot.bar(title= 'Survived')
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
plt.figure(1) 
plt.subplot(221) 
train_data['Sex'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Sex')   
plt.subplot(222) 
train_data['Cabin'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Cabin') 
plt.show()
plt.figure(1) 
plt.subplot(131) 
train_data['Pclass'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Pclass') 
plt.subplot(132) 
train_data['SibSp'].value_counts(normalize=True).plot.bar(title= 'SibSp') 
plt.subplot(133) 
train_data['Parch'].value_counts(normalize=True).plot.bar(title= 'Parch') 
plt.show()
plt.figure(1) 
plt.subplot(131) 
train_data['Embarked'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Embarked') 
plt.show()
import seaborn as sns                  # For data visualization 
plt.figure(1) 
plt.subplot(121) 
df=train_data.dropna() 
sns.distplot(train_data['Age']); 
plt.subplot(122)
train_data['Age'].plot.box(figsize=(16,5))
plt.show()
plt.figure(1) 
plt.subplot(121) 
df=train_data.dropna() 
sns.distplot(train_data['Fare']); 
plt.subplot(122)
train_data['Fare'].plot.box(figsize=(16,5))
plt.show()
Sex=pd.crosstab(train_data['Sex'],train_data['Survived']) 
Sex.div(Sex.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Cabin=pd.crosstab(train_data['Cabin'],train_data['Survived']) 
Cabin.div(Cabin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Pclass=pd.crosstab(train_data['Pclass'],train_data['Survived']) 
Pclass.div(Pclass.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
SibSp=pd.crosstab(train_data['SibSp'],train_data['Survived']) 
SibSp.div(SibSp.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Parch=pd.crosstab(train_data['Parch'],train_data['Survived']) 
Parch.div(Parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Embarked=pd.crosstab(train_data['Embarked'],train_data['Survived']) 
Embarked.div(Embarked.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
train_data['Total_Relative']=train_data['SibSp']+train_data['Parch']
bins=[0,2,4,6,81] 
group=['Low','Average','High', 'Very high'] 
train_data['Total_Relative_bin']=pd.cut(train_data['Total_Relative'],bins,labels=group)
Total_Relative_bin=pd.crosstab(train_data['Total_Relative_bin'],train_data['Survived']) 
Total_Relative_bin.div(Total_Relative_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Relative') 
P = plt.ylabel('Percentage')
train_data.groupby('Survived')['Age'].mean().plot.bar()
train_data.groupby('Survived')['Fare'].mean().plot.bar()
bins=[0,25,40,60,810] 
group=['Low','Average','High', 'Very high'] 
train_data['Age_bin']=pd.cut(train_data['Age'],bins,labels=group)
Age_bin=pd.crosstab(train_data['Age_bin'],train_data['Survived']) 
Age_bin.div(Age_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Age') 
P = plt.ylabel('Percentage')
bins=[0,25,40,60,810] 
group=['Low','Average','High', 'Very high'] 
train_data['Fare_bin']=pd.cut(train_data['Fare'],bins,labels=group)
Fare_bin=pd.crosstab(train_data['Fare_bin'],train_data['Survived']) 
Fare_bin.div(Fare_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Fare') 
P = plt.ylabel('Percentage')
train_data.columns
train_data.shape
train_data=train_data.drop(['Fare_bin', 'Age_bin', 'Total_Relative_bin', 'Total_Relative'], axis=1)
matrix = train_data.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train_data.isnull().sum()
train_data['Cabin'].fillna(train_data['Cabin'].mode()[0], inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data.isnull().sum()
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data.isnull().sum()
test_data['Cabin'].fillna(test_data['Cabin'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data.isnull().sum()
train_data['Age_log'] = np.log(train_data['Age'])
train_data['Age_log'].hist(bins=20)
test_data['Age_log'] = np.log(test_data['Age'])
train_data['Fare_log'] = np.log(train_data['Fare'])
train_data['Fare_log'].hist(bins=20)
test_data['Fare_log'] = np.log(test_data['Fare'])
train_data=train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
test_data=test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
x = train_data.drop('Survived',1)
y = train_data.Survived
x=pd.get_dummies(x)
train_data=pd.get_dummies(train_data)
test_data=pd.get_dummies(test_data)
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x,y, test_size =0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='12', random_state=1, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
pred_cv = model.predict(x_cv)