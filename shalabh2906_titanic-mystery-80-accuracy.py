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
#TitanicPrdiction

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Extracting Data

df_train_dataset=pd.read_csv('../input/titanic/train.csv')

df_test_dataset=pd.read_csv('../input/titanic/test.csv')

df_gender_submission=pd.read_csv('../input/titanic/gender_submission.csv')
df_train_dataset.info()
#Concatinating Train and Test Dataset

df_complete_data=pd.concat([df_train_dataset,df_test_dataset])
#Data Preperation

df_complete_data.describe()
#Survived, Age, Cabin, Fare, embarked have null values

df_complete_data.info()
#Total Dataset

df_complete_data.shape
#Changing Fare Null Values

df_complete_data['Fare'].isnull().sum()
#Creating Median Value 

Med_value=df_complete_data[df_complete_data['Pclass']==3]['Fare'].median()

df_complete_data.loc[df_complete_data['Ticket']=='3701',['Fare']]=Med_value
#Fare null value changed and Filled

df_complete_data['Fare'].isnull().sum()
#Checking Null values in Embarked

df_complete_data['Embarked'].isnull().sum()
df_complete_data.loc[df_complete_data['Ticket']=='113572',['Embarked']]='S'
df_complete_data['Ticket'].isnull().sum()
#All are false means data is filled

df_complete_data.loc[df_complete_data['Ticket']=='113572'].isnull().describe()
# Checking Age for Null values

df_complete_data['Age'].isnull().sum()
#Segregating Unique titles

df_complete_data['Title']=df_complete_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_complete_data['Title'].unique()
df_complete_data['Title'].value_counts()
#categorizing Name on the basis of age and Sex

for i in range (len(df_complete_data)):

    if ((df_complete_data.iloc[i,df_complete_data.columns.get_loc('Sex')]=='male') and (df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]<=14)):

        df_complete_data.iloc[i, df_complete_data.columns.get_loc('Title')] = 1

    if ((df_complete_data.iloc[i,df_complete_data.columns.get_loc('Sex')]=='female') and (df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]<=14)):

        df_complete_data.iloc[i, df_complete_data.columns.get_loc('Title')] = 2

    if ((df_complete_data.iloc[i,df_complete_data.columns.get_loc('Sex')]=='male') and (df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]>14)):

        df_complete_data.iloc[i, df_complete_data.columns.get_loc('Title')] = 3

    if ((df_complete_data.iloc[i,df_complete_data.columns.get_loc('Sex')]=='female') and (df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]>14)):

        df_complete_data.iloc[i, df_complete_data.columns.get_loc('Title')] = 4

        
#Checking Null Value in Age 

df_complete_data[df_complete_data['Age'].isnull()]
Mean_1=df_complete_data[df_complete_data['Title']==1]['Age'].mean()

Mean_2=df_complete_data[df_complete_data['Title']==2]['Age'].mean()

Mean_3=df_complete_data[df_complete_data['Title']==3]['Age'].mean()

Mean_4=df_complete_data[df_complete_data['Title']==4]['Age'].mean()

print(Mean_1)

print(Mean_2)

print(Mean_3)

print(Mean_4)
#Taking mean values 6 and 32

for i in range (len(df_complete_data)):

    if df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')] in ['Mr','Miss','Mrs','Rev','Dr','Col','Ms','Mlle','Major','Jonkheer','Don','Mme','Dona','Lady','the Countess','Capt','Sir']:

        df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]=32

    if df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')] in ['Master']:

        df_complete_data.iloc[i,df_complete_data.columns.get_loc('Age')]=6

    if df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')] in ['Dr','Capt','Col','Don','Jonkheer','Major','Mr','Rev','Sir']:

        df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')]=3

    if df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')] in ['Dona','Lady','Miss','Mlle','Mme','Mrs','Ms']:

        df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')]=4

    if df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')] in ['Master']:

        df_complete_data.iloc[i,df_complete_data.columns.get_loc('Title')]=1
df_complete_data['Age'].isnull().sum()
df_complete_data.drop(columns=['Cabin','Name','Ticket'], inplace=True)

df_complete_data.info()
#mapping the data

cleanup_maps={"Sex " :  {"male" : 1, "female" :  2},

             "Emabarked " :  {'S' : 1, 'Q' : 2, 'C' : 3}}
df_complete_data.replace(cleanup_maps, inplace=True)
df_complete_data.head()
df_complete_data.iloc[0]
df_trainset=df_complete_data.iloc[0:891,:]

df_testset=df_complete_data.iloc[891:,:]

print(df_trainset.shape)

print(df_testset.shape)
train_passengerId=df_trainset['PassengerId']

test_passengerId=df_testset['PassengerId']

target=df_trainset['Survived']

df_trainset.drop(columns=['PassengerId','Survived'],inplace=True)

df_testset.drop(columns=['PassengerId','Survived'],inplace=True)
df_trainset.info()
df_testset.info()
# Calling Decision Tree Model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
one_hot_encoded_training_predictors = pd.get_dummies(df_trainset)
one_hot_encoded_testing_predictors = pd.get_dummies(df_testset)
#logistic_regression= LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=6)

decision_tree = decision_tree.fit(one_hot_encoded_training_predictors, target)

pred=decision_tree.predict(one_hot_encoded_testing_predictors)
pred
data = {'PassengerId':test_passengerId, 'Survived':pred}

Result=pd.DataFrame(data)

Result['Survived']=Result['Survived'].apply(lambda x: int(x))

Result.head(25)
Result.to_csv('Result.csv', index=False)