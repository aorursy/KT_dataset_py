# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# for visualization of various graph



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic_train.csv')

test=pd.read_csv('../input/titanic_test.csv')
train.head()

#to have certain information about data
train.columns
train.info()

train.describe()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='summer')
train
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='ocean')
sns.countplot(x='Survived',hue='Sex',data=train,palette='ocean')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='Blues')
sns.distplot(train['Age'].dropna(),kde=False,color='purple',bins=35)
sns.countplot(x='SibSp',data=train, color='purple')
train['Fare'].hist(color='purple',bins=40,figsize=(8,4))
plt.figure(figsize=(13,8))

sns.swarmplot(x="Age", y='Fare', data=train, hue='Survived')
sns.jointplot(x='Survived', y='Age', data=train, kind='hex', color='purple')
sns.jointplot(x='Survived', y='Fare', data=train, kind='reg', color='purple')
import cufflinks as cf

cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='blue')
train['Age'].iplot(kind='hist',bins=30,color='blue')
fig, ax = plt.subplots(figsize=(7,7)) 

sns.heatmap(train[['Survived', 'Pclass', 'Sex', 'SibSp', 'Embarked', 'Age', 'Fare']].corr(method='pearson'), 

            annot = True, square=True, fmt='.2g', vmin=-1, vmax=1, center= 0, cmap= 'ocean', ax=ax, linewidths=.5, cbar=False)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='ocean')
def impute(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Blues')
plt.figure(figsize=(13,8))

sns.swarmplot(x="Age", y='Fare', data=train, hue='Survived')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
train
test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
predictions.shape
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions))

test.head()
test.info()

test.columns
train.corr()
plt.figure(figsize=(13,9))

sns.heatmap(train.corr(),cmap='ocean',annot=True)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train, y_train)
lm.intercept_
lm.coef_
prediction= lm.predict(X_test)
prediction
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print(np.sqrt(metrics.mean_squared_error(y_test, prediction)))
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]
features = ["Pclass",  "SibSp", "Parch"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])





model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictionss = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictionss})

output.to_csv('Submission.csv', index=False)
