# Import necessary packages

import numpy as np 

import pandas as pd 

import seaborn as sb

import matplotlib.pyplot as plt

% matplotlib inline

import warnings

warnings.simplefilter("ignore")

import os

print(os.listdir("../input"))

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.metrics import explained_variance_score

from xgboost import XGBClassifier
# Load train dataset

train=pd.read_csv('../input/train.csv')

train.head()
# Load test dataset

test=pd.read_csv('../input/test.csv')

test.head()
train.info()
# Drop uneccessary columns

train.drop(columns=['PassengerId','Name','Cabin','Ticket','Fare'],inplace=True)

test.drop(columns=['Name','Ticket','Cabin','Fare'],inplace=True)
# Get index with null values in train dataset

index_list=train[train['Age'].isnull()].index

index_list
# Fill those null values with appropiate mean values

for index in index_list:

    if train.loc[index,'Pclass']==1 and train.loc[index,'Sex']=='female':

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[1][0])

    elif train.loc[index,'Pclass']==1 and train.loc[index,'Sex']=='male':

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[1][1])

    elif train.loc[index,'Pclass']==2 and train.loc[index,'Sex']=='female':

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[2][0])

    elif train.loc[index,'Pclass']==2 and train.loc[index,'Sex']=='male':

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[2][1])

    elif train.loc[index,'Pclass']==3 and train.loc[index,'Sex']=='female':

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[3][0])

    else:

        train.loc[index,'Age']=np.ceil(train.groupby(['Pclass','Sex'])['Age'].mean()[3][1])
# Fill Embarked with mode of the column

train['Embarked'].fillna(train['Embarked'][0],inplace=True)
# Get index with null values in test dataset

index_list=test[test['Age'].isnull()].index

index_list
# Fill those null values with appropiate mean values

for index in index_list:

    if test.loc[index,'Pclass']==1 and test.loc[index,'Sex']=='female':

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[1][0])

    elif test.loc[index,'Pclass']==1 and test.loc[index,'Sex']=='male':

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[1][1])

    elif test.loc[index,'Pclass']==2 and test.loc[index,'Sex']=='female':

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[2][0])

    elif test.loc[index,'Pclass']==2 and test.loc[index,'Sex']=='male':

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[2][1])

    elif test.loc[index,'Pclass']==3 and test.loc[index,'Sex']=='female':

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[3][0])

    else:

        test.loc[index,'Age']=np.ceil(test.groupby(['Pclass','Sex'])['Age'].mean()[3][1])
# Check if the above operations worked correctly

train.isnull().sum().max(),test.isnull().sum().max()
base_color=sb.color_palette()[0]
# Bivariate plot of Survived vs. Age

sb.distplot(train[train['Survived']==1]['Age'],label='Survived');

sb.distplot(train[train['Survived']==0]['Age'],label='Not Survived');

plt.legend();

plt.title('Survived vs. Age');
# Multi-variate plot of Survived vs Age by Gender

sb.pointplot(data=train,x='Survived',y='Age',hue='Sex',linestyles="",dodge=0.3);

xticks=[0,1]

xlabel=['No','Yes']

plt.xticks(xticks,xlabel);

plt.title('Survived vs Age by Gender');
# Multi-variate plot of Survived vs Age by Class

sb.pointplot(data=train,x='Survived',y='Age',hue='Pclass',linestyles="",dodge=0.3,palette='viridis_r');

xticks=[0,1]

xlabel=['No','Yes']

plt.xticks(xticks,xlabel);

plt.title('Survived vs Age by Class');
'''

single=[train,test]

# Map columns to numerical values

for data in single:

    data['Sex']=data['Sex'].map({'female':1,'male':0}).astype(int)

    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)

'''
# Merge the two datasets

ntrain = train.shape[0]

ntest = test.shape[0]

all_data = pd.concat((train, test))
# Get dummy variables

all_data=pd.get_dummies(all_data)
# Seperate the combined dataset into test and train data

test=all_data[all_data['Survived'].isnull()]

train=all_data[all_data['PassengerId'].isnull()]
# Check if the new and old sizes are equal

assert train.shape[0]==ntrain

assert test.shape[0]==ntest
# Drop extra columns

test.drop(columns='Survived',inplace=True)

train.drop(columns='PassengerId',inplace=True)

test['PassengerId']=test['PassengerId'].astype(int)
# Divide the data into test and train

X_train=train.drop('Survived',axis=1)

Y_train=train['Survived']

X_test=test.drop('PassengerId',axis=1)
'''

# Fit the model using Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

'''
# Fit the model using XGBClassifier

xgb = xgboost.XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)

xgb.fit(X_train,Y_train)

Y_pred = xgb.predict(X_test)
final_df = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })
final_df['Survived']=final_df['Survived'].astype(int)
# Save the dataframe to a csv file

final_df.to_csv('submission.csv',index=False)