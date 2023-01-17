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
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.shape
train.head(20)
train.describe(include = 'all')
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=True)
sns.countplot(x= 'Survived', data = train  )
sns.countplot(x= 'Sex',hue = 'Survived', data = train  )
sns.countplot(x= 'Embarked',hue = 'Survived', data = train  )
sns.countplot(x= 'Pclass',hue = 'Survived', data = train  )
sns.catplot(x= 'SibSp',y = 'Survived',kind = 'bar', data = train  )
sns.catplot(x= 'Parch',y = 'Survived',kind = 'bar', data = train  )
sns.catplot(x= 'Pclass',y = 'Embarked',hue = 'Survived',kind = 'violin', data = train  )
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=15)
g = sns.FacetGrid(train, col='Survived',row = 'Pclass')
g.map(plt.hist, 'Fare', bins=15)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
sns.catplot(y = 'Pclass',  x = 'Embarked', kind = 'bar',col = 'Survived',row = 'Sex' ,data = train)

train.isnull().sum()
sns.heatmap(train.isnull())
train_df = train.drop(['Ticket', 'Cabin'], axis=1)
test_df = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df] #Combinning both the table otherwise we have do thesame things twice.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #it will give the strings which is end up with '.' .

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean() 

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
#Let's see the relation betwwen title and survived columns.
sns.barplot(y = 'Survived', x = 'Title', data = train_df)
#Now safely delete the Name Name from both of the dataset only PassengerId from the train Dateset.
train_df = train_df.drop(['Name','PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
combine =[train_df,test_df]
#Let's handle the null values.
train_df.isnull().sum()
#test['Fare'].fillna(test['Fare'].mean(),inplace = True)
# An example of visualizing the outliers.
sns.boxplot( x = 'Age', data = train_df)
# detecting outliers

Q1 = train_df.quantile(.25)
Q3 = train_df.quantile(.75)
IQR = Q3 - Q1
print(IQR)

#for the test data
Q1 = test_df.quantile(.25)
Q3 = test_df.quantile(.75)
IQR = Q3 - Q1
print(IQR)

print(train_df < (Q1 -1.5 * IQR)) or (train_df > (Q3  + 1.5 * IQR))
#Replace the outliers of age columns with the median value
print(train_df['Age'].quantile(.50))
print(train_df['Age'].quantile(.95))
print(train_df['Age'].quantile(.75))

#for the test data

test_df['Age'].quantile(.50)
test_df['Age'].quantile(.95)
test_df['Age'].quantile(.75)

train_df['Age'] = np.where(train_df['Age']>54.0, 35.0, train_df['Age'])


# test data
test_df['Age'] = np.where(test_df['Age']>54.0, 35.0, test_df['Age'])
Combine = [train_df , test_df]
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['Embarked'] = train_df['Embarked'].fillna('S')
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())  
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())        
test_df.isnull().sum()
#Earlier we seen that catagory of age has a great impact on the Survival .So let's make some Age group
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()
#Let' create a new features from wxixsting SibSp and Parch.
for dataset in combine:
     dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
     dataset.loc[dataset['FamilySize'] > 1, 'FamilySize'] = 1
     dataset.loc[dataset['FamilySize'] < 0, 'FamilySize'] = 0
    
train_df.head()
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]


train_df['FareBand'] = pd.cut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
   
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
train_df.head()
X = train_df.iloc[:,1:8]
y = train_df.Survived
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

rm = RandomForestClassifier()
rm.fit(X_train,y_train)
rm.score(X_test,y_test)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
import xgboost as xgb
train = xgb.DMatrix(X_train, label = y_train)
test = xgb.DMatrix(X_test, label = y_test)
from xgboost import XGBClassifier
XGBClassifier()
model = XGBClassifier(max_depth = 1,
                     n_estimators = 100,
                     learning_rate = 1,
                     min_child_weight = 1,
                     random_state = 45,
                     reg_alpha = 0
                     )

           
x = model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)

Pid = test_df['PassengerId']
pred = model.predict(test_df.drop(['PassengerId'],axis = 1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : Pid, 'Survived': pred })
output.to_csv('submission.csv', index=False)
