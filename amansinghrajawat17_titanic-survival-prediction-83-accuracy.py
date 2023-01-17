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

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('whitegrid')
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

Passenger_ID = test['PassengerId']
sns.heatmap(train.isnull(), cmap='plasma')
sns.countplot('Survived', data = train, hue='Sex')
sns.countplot('SibSp', data = train)
train.sort_values(by=['SibSp'], ascending=False).head(10)
outliner_SibSp = train.loc[train['SibSp'] == 8]
outliner_SibSp
train = train.drop(outliner_SibSp.index, axis=0)
train.sort_values(by=['Fare', 'Pclass'], ascending=False)
sns.boxplot(train['Fare'],orient='v')
outliner_Fare = train.loc[train['Fare']>500]
outliner_Fare
train = train.drop(outliner_Fare.index, axis=0)
dataset = pd.concat([train, test], ignore_index=True)
dataset.head()
sns.heatmap(dataset.isnull(), cmap='viridis')
dataset.shape
dataset = dataset.fillna(np.nan)

dataset.isnull().sum()
dataset.loc[dataset['Embarked'].isnull()]
sns.countplot(dataset['Embarked'])
dataset['Embarked'] = dataset['Embarked'].fillna('S')
dataset.loc[dataset['Fare'].isnull()]
temp = dataset[(dataset['Pclass'] == 3) & (dataset['Parch'] ==0) & (dataset['SibSp'] == 0) 

               & (dataset['Fare']>0)].sort_values(by='Fare', ascending=False)
temp.mean()
temp['Fare'].mean()
dataset['Fare'] = dataset['Fare'].fillna(temp['Fare'].mean())
dataset.isnull().sum()
dataset[(dataset['Survived'] == 0) & (dataset['Sex'] == 'male')]['Age'].mean()
nullAge = dataset.loc[dataset['Age'].isnull()]
nullAge.shape
def impute_age(cols):

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
dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age, axis=1)
dataset.isnull().sum()
sns.catplot(data = dataset, x = 'Pclass', y = 'Survived', kind='bar')
g = sns.FacetGrid(data = dataset[dataset['Survived'] == 1], col='Pclass')

g.map(sns.countplot, 'Sex')
X=dataset.drop(['Cabin','Name','PassengerId','Survived','Ticket'],axis=1)

Y=dataset['Survived']
X.head(3)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
X['Embarked']=LabelEncoder().fit_transform(X['Embarked'])

X['Sex']=LabelEncoder().fit_transform(X['Sex'])

X['Age']=StandardScaler().fit_transform(np.array(X['Age']).reshape(-1,1))

X['Fare']=StandardScaler().fit_transform(np.array(X['Fare']).reshape(-1,1))
X.head(3)
from sklearn.model_selection import train_test_split
trainDataX=X[:train.shape[0]]

trainDataY=Y[:train.shape[0]].astype('int32')

testDataX=X[train.shape[0]:]
X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.1,random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier().fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_train, dtree.predict(X_train)))
print(accuracy_score(Y_test, dtree.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier().fit(X_train, Y_train)
print(accuracy_score(Y_train, randomForest.predict(X_train)))
print(accuracy_score(Y_test, randomForest.predict(X_test)))
submission = pd.DataFrame(columns=['PassengerId','Survived'])

submission['PassengerId'] = Passenger_ID

submission['Survived'] = dtree.predict(testDataX)
submission.head()
filename = 'submit.csv'
submission.to_csv(filename, index=False)
from IPython.display import FileLink

FileLink(filename)