#Import all the necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

%matplotlib inline
#load the train data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.head()

test.head()
train.info()
test.info()
train.describe(include = 'all') # for statistical information which is not for categorical data
# Lets check for other missing data

train.isna().sum()

test.isna().sum()
sns.countplot(x = 'Survived',hue = 'Sex',data = train,palette = 'magma')
sns.countplot(x = 'Survived',hue = 'Pclass',data = train,palette = 'tab20b_r')
sns.set_style('whitegrid')

train['Age'].plot(kind = 'hist',bins = 40,color = 'red')
sns.barplot(x='SibSp',y='Survived',data=train)
# First we will have a loot at test data also

test.describe(include='all')

train.describe(include = 'all')
# Visualize the missing data 

sns.heatmap(train.isnull(),yticklabels  = False,cbar = False,cmap = 'viridis_r')
sns.heatmap(test.isnull(),yticklabels = False,cbar  =False,cmap = 'viridis')
# Age Feature

train['Age'] = train['Age'].fillna(train['Age'].mean()) # fill for train

test['Age'] = test['Age'].fillna(test['Age'].mean()) # fill for test 
train['Embarked'].value_counts() # Group Wise count records
# Fill to Embarked column NA with S

train['Embarked'] = train['Embarked'].fillna('S')
#We will fill the Na in test data for 'Fare' column

test['Fare']=test["Fare"].fillna(test["Fare"].mean())
# We will drop the name column, as it has no much of significance

train.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)

test.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)
# we will creat a dummy varibale for sexs Column

sex = pd.get_dummies(train['Sex'],drop_first=1)

sex1 = pd.get_dummies(test['Sex'],drop_first = 1)
# we will creat a dummy variable for Embarked column

embark  = pd.get_dummies(train['Embarked'],drop_first = 1)

embark1 = pd.get_dummies(test['Embarked'],drop_first = 1)
# we will check the head of my dummy

sex.head()

embark.head()
# we will concatenat sex and embark dummy variables into train datasets 

train = pd.concat([train,sex,embark],axis = 1)
# we will concatenat sex1,embark1 dummy varibales into test datasets

test = pd.concat([test,sex1,embark1],axis = 1)
# we will check the head of my datasets

train.head()

test.head()
# we will drop the Sex and Embarked columns from train datsets,as it has no much of significance

train.drop(['Sex','Embarked'],axis =1,inplace =True)
# we will also drop the Sex and Embarked columns from test datsets,as it has no much of significance

test.drop(['Sex','Embarked'],axis = 1,inplace = True)
# now we will check head of my datasets

train.head()

test.head()
# Plot Counts for Each survived groupby counts



fig = px.bar(train.groupby('Survived').count())

fig.show()
fig = px.bar(train.groupby('male').mean())

fig.show()
# Parch and Survived Bar graph

plt.figure(figsize = (12,8))

sns.barplot(x = 'Parch',y = 'Survived',data = train)
plt.figure(figsize = (12,8))

px.bar(train,x = 'Parch',y = 'Survived',color = 'male')
#  Pclass wise survived graph 



px.bar(train,x = 'Pclass',y ='Survived',color = 'Pclass')
# Gender wise Survived graph 

# we get 1 for female and 0 for male



fig = px.bar(train, x='male', y='Survived', color='male')

fig.show()

 
# show correlations Heatmap 

plt.figure(figsize = (12,8))

sns.heatmap(train.corr(),annot = True,cmap = 'PiYG_r')
# we will predict the family size of passengers from train datasets

train['Family_size'] = train['SibSp']+train['Parch']+1
# we will predict the family size of passengers from test datasets

test['Family_size'] = test['SibSp']+test['Parch']+1
# we will check the head of my datasets

train.head()

test.head()
# we will creat a functions to predict the family_group 

def family_group(size):

    a = ''

    if (size<=1):

        a = 'alone'

    elif(size<=4):

        a = 'small'

    else:

        a = 'large'

    return a
# now we will map family group  functions to train datasets 

train['Family_Group'] = train['Family_size'].map(family_group)
# we will also map family group functions to test datasets

test['Family_Group']  = test['Family_size'].map(family_group)
# now we will check the head of my datasets

train.head()

test.head()
# we will creat a functions to predict the age group

def age_group(ages):

    a = ''

    if (ages<= 1):

        a = 'infant'

    elif (ages<=4):

        a = 'baby'

    elif (ages<=14):

        a = 'child'

    elif (ages<=19):

        a = 'teenager'

    elif (ages<=24):

        a = 'young adult'

    elif(ages<=35):

        a = 'adult'

    elif(ages<=50):

        a = 'senior'

    else:

        a ='old'

    return a

        
# now we will map age group  functions to train datasets 

train['Age_group'] = train['Age'].map(age_group)
# now we will map age group  functions to test datasets 

test['Age_group'] = test['Age'].map(age_group)
# now will check the head of my datasets

train.head()

test.head()
# we will creat a functions to predict the fare group

def fare_group(fare):

    a = ''

    if fare<=4:

        a ='very low'

    elif fare<=10:

        a = 'low'

    elif fare<=20:

        a = 'mid'

    elif fare<= 45:

        a = 'High'

    else:

        a = 'Very High'

    return a
#  we will map fare group  functions to train datasets 

train['Fare_Group'] = train['Fare'].map(fare_group)
#  we will also map age fare group to test datasets 

test['Fare_Group'] = test['Fare'].map(fare_group)
# now we will check the head of my datasets

train.head()

test.head()
#  we will creat a dummy varibales for family_group,age_group,fare_group Columns

group = pd.get_dummies(train[['Family_Group','Age_group','Fare_Group']],drop_first = 1)
# we will also creat a dummy varibales for family_group,age_group,fare_group Columns

group1 = pd.get_dummies(test[['Family_Group','Age_group','Fare_Group']],drop_first = 1)
# we will check the head of my newly created dummy varibales

group.head()

group1.head()
# we will concatenat the group(dummy variables) into train datasets 

train = pd.concat([train,group],axis = 1)
# we will concatenat the group1(dummy variables) into test datasets 

test = pd.concat([test,group1],axis = 1)
# now will check the head of my datasets one more time

train.head()

test.head()
# we will drop the some columns from train datasets,as it has no much of significance

train.drop(['Family_Group','Age_group','Fare_Group','Age','Fare','Family_size','SibSp','Parch'],axis = 1,inplace =True)
# we will also drop the some columns from test datasets ,as it has no much of significance

test.drop(['Family_Group','Age_group','Fare_Group','Age','Fare','Family_size','SibSp','Parch'],axis = 1,inplace =True)
# now we check the head of my datasets last time

train.head()

test.head()
X = train.drop(['Survived','PassengerId'],axis = 1)

y = train['Survived']



X.shape,y.shape
#Lets split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report

print(accuracy_score(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
# we will try Random Forest for better accuracy

from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)

rfc_preds= rfc.predict(X_test)



print(accuracy_score(y_test,rfc_preds))

print('\n')

print(classification_report(y_test,rfc_preds))

from sklearn.svm import SVC



svc=SVC()

svc.fit(X_train,y_train)

svc_preds=svc.predict(X_test)



from sklearn.metrics import accuracy_score



print(accuracy_score(y_test,svc_preds))

print('\n')

print(classification_report(y_test,svc_preds))
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

gbk_preds = gbk.predict(X_test)

print(accuracy_score(y_test,gbk_preds))

print('\n')

print(classification_report(y_test,gbk_preds))
passenger_id = test['PassengerId']

predictions = logmodel.predict(test.drop('PassengerId',axis = 1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : passenger_id, 'Survived': predictions })

output.to_csv('submission_1.csv', index=False)