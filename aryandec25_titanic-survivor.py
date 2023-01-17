import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os       

print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train.shape
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Lets have a countplot based on the Survived column

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
#Lets have the same countplot when compared to gender wise.

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
#Let us categorize the data

num_cols=[var for var in train.columns if train[var].dtypes != 'O']

cat_cols=[var for var in train.columns if train[var].dtypes != 'int64' and train[var].dtypes != 'float64']

print('No of numerical cols: ',len(num_cols))

print('No of categoriacl cols: ',len(cat_cols))

print('Total No of Cols: ',len(cat_cols+ num_cols))
#Lets create a histogram for the age column

train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sum(train['PassengerId'].duplicated()),sum(test['PassengerId'].duplicated())
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
#Columns with null values in the Train dataFrame

var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]



for var in var_with_na:

    print(var, np.round(train[var].isnull().mean(),3), '% missing values')
#Columns with null values in the Test dataFrame

var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]



for var in var_with_na2:

    print(var, np.round(test[var].isnull().mean(),3), '% missing values')
test['Fare'].head()
num_miss_vars=['age','Fare']

cat_miss_vars=['Embarked']

drop_cols=['Cabin']
combine=[train,test]



for df in combine:

    df.drop(columns=drop_cols, inplace=True)
for df in combine:

    df['Embarked'].fillna('S',inplace=True)
test['Fare'].fillna('35.6',inplace=True)
#Lets create a heatmap to see which all columns has null values

sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age

    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] =train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.head()
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
test.head()
X_train=train.drop(columns='Survived')

Y_train=train['Survived']

#X_test=test.drop(columns='PassengerId')

X_test1=test
#X_test1['Fare']=X_test1['Fare'].astype('float64')
#from sklearn.linear_model import LogisticRegression

#logmodel = LogisticRegression()

#logmodel.fit(X_train,Y_train)

#predictions = logmodel.predict(X_test1)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test1)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
final_df = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })
# Save the dataframe to a csv file

final_df.to_csv('submission.csv',index=False)