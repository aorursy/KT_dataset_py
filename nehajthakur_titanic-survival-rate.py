# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

train_df.head()
print(train_df.columns)






#  passing by reference is convenient, because we can clean both datasets at once

combine = [train_df,test_df]

train_df[['Pclass',"Survived"]].groupby(['Pclass']).mean().sort_values(by = 'Survived', ascending=False)
train_df[['Sex','Survived']].groupby('Sex').mean().sort_values('Survived', ascending=False)
train_df[['SibSp','Survived']].groupby('SibSp').mean().sort_values('Survived',ascending =False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Age',bins=30);
grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', height=2.2, aspect =1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins=20 )

grid.add_legend()
# the x category is the Pclass and the hue category is the Sex. Hence you need to add

# order = [1,2,3], hue_order=["male", "female"]



grid = sns.FacetGrid(train_df, row = 'Embarked', height = 2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass','Survived','Sex', order = [1,2,3], hue_order=["female","male"],palette = 'deep')

grid.add_legend()

plt.show()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height =2.2, aspect=1.6 )

grid.map(sns.barplot, 'Sex','Fare', alpha=0.5, ci=None, order = ['male',"female"])

grid.add_legend();
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby('Title').mean()
title_mapping = {'Mr':1,'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}



for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()  
train_df = train_df.drop(["PassengerId","Name"],axis= 1)

test_df = test_df.drop(['Name'], axis=1)



combine = [train_df, test_df]



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
### COMPLETING or replacing the NAN values with relevant values



for dataset in combine:

    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)

    

    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

    

    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    



print("Training data with null values per column: \n",train_df.isnull().sum())

print("\n")



print("testing data with null values per column: \n", test_df.isnull().sum())





train_df['Age'] = train_df['Age'].astype(int)

train_df['Age'] = train_df['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'],5 )

train_df[['AgeBand','Survived']].groupby('AgeBand',as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[dataset['Age'] <=16, 'Age'] =0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df =train_df.drop(['AgeBand'],axis = 1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1

    

train_df[["FamilySize","Survived"]].groupby(["FamilySize"],as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize']==1, 'IsAlone']=1



train_df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index = False).mean()
train_df = train_df.drop(["Parch","SibSp", "FamilySize"],axis=1)

test_df = test_df.drop(["Parch","SibSp", "FamilySize"],axis=1)



combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset["Age*Class"] = dataset.Age * dataset.Pclass

    

train_df[["Age*Class", "Age","Pclass"]].head()    
for dataset in combine:

    dataset['Embarked'] = dataset.Embarked.map({'S': 0, 'C': 1, 'Q': 2}) .astype(int)

    

train_df.head()    
train_df["FareBand"] = pd.qcut(train_df['Fare'],4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).count().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
# using Logistic regression

X_train = train_df.drop('Survived',axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId',axis=1).copy()

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

y_predict = logreg.predict(X_test)

logistics_regression_acc_log = round(logreg.score(X_train,Y_train)*100,2)

logistics_regression_acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],

                           "Survived":y_predict })

submission.to_csv('submission.csv', index=False)