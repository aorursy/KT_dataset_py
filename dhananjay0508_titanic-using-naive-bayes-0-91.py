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
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar = False,cmap = 'viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar = False,cmap = 'viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data = train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue = 'Sex',data = train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue = 'Pclass',data = train)
sns.distplot(train['Age'].dropna(),kde = False,color = 'darkred',bins = 40)
sns.countplot(x= 'SibSp',data=train)
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(train.isnull(),yticklabels=False,cbar = False,cmap = 'viridis')
train.isnull().sum()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(test.isnull(),yticklabels=False,cbar = False,cmap = 'viridis')
del test['Cabin']
del train['Cabin']
test.isnull().sum()
train.isnull().sum()
train.dropna(inplace = True)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train.info()
test.info()
sex = pd.get_dummies(train['Sex'],drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'],axis = 1,inplace = True)
train = pd.concat([train,sex,embark],axis = 1)
train.head()
sex = pd.get_dummies(test['Sex'],drop_first = True)
embark = pd.get_dummies(test['Embarked'],drop_first = True)
test.drop(['Sex','Embarked','Name','Ticket'],axis = 1,inplace = True)
test = pd.concat([test,sex,embark],axis =1)
test.head()
train.columns
features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'male', 'Q', 'S']
X= train[features].copy()
X.head()
y = train['Survived']
y.head()
X_test = test[features]
X_test.head()
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,y)
predictions = clf.predict(X_test)
df = pd.read_csv('../input/titanic/gender_submission.csv')
df.columns
y_test = df['Survived']
accuracy_score(y_true = y_test, y_pred = predictions)
output = pd.DataFrame({
    "PassengerId": X_test['PassengerId'],
    "Survived": predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

