# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info
test.info
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))             
train.Survived.value_counts().plot(kind='bar')# bar graph 
plt.title(u"survival counts (1=survived)") #name
plt.ylabel(u"numbers")
plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"numbers")
plt.title(u"cabin classes")
plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived, train.Age)
plt.ylabel(u"age")                         
plt.grid(b=True, which='major', axis='y') 
plt.title(u"survival distribution (1=survived)")

plt.subplot2grid((2,3),(1,0), colspan=2)
train.Age[train.Pclass == 1].plot(kind='kde')   
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"age distribution of different class")
plt.legend((u'class1', u'class2',u'class3'),loc='best')
plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.title(u"population from differnt piers")
plt.ylabel(u"numbers")  
plt.show()
fig = plt.figure()
fig.set(alpha=0.2)  #set color for the graph
Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({u'being rescued':Survived_1, u'not rescued':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"being rescued or not")
plt.xlabel(u"passengers' classes") 
plt.ylabel(u"numbers") 
fig = plt.figure()
fig.set(alpha=0.2)  # set color for the graph
 
Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df=pd.DataFrame({u'survivor':Survived_1, u'vistim':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Survival rate of passengers from differnt ports")
plt.xlabel(u"ports") 
plt.ylabel(u"numbers") 
 
plt.show()
fig.set(alpha=0.2)  # set color for the graph
Survived_m = train.Survived[train.Sex == 'male'].value_counts()
Survived_f = train.Survived[train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"survival analysis according to gender ")
plt.xlabel(u"gender") 
plt.ylabel(u"number")
fig = plt.figure()
fig.set(alpha=0.2)  
 
Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df=pd.DataFrame({u'survivor':Survived_1, u'victims':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Survival rate of differnt piers")
plt.xlabel(u"piers") 
plt.ylabel(u"numbers") 
 
plt.show()
print((train.Fare == 0).sum())
import numpy as np
train.Fare = train.Fare.replace(0, np.NaN)
print ((train.Fare == 0).sum())
train[train.Fare.isnull()].index
train.Fare.mean()
train.Fare.fillna(train.Fare.mean(),inplace=True)
train[train.Fare.isnull()]
print((train.Age == 0).sum())#no need to impute
train.Age.fillna(train.Age.mean(),inplace=True)
train[train.Age.isnull()]
train.Cabin.isnull().mean() # no need to impute
train['Name_len']=train.Name.str.len()
train['Ticket_First']=train.Ticket.str[0]
train['FamilyCount']=train.SibSp+train.Parch
train['Cabin_First']=train.Cabin.str[0]
# Regular expression to get the title of the Name
train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)
train.title.value_counts().reset_index()
train.columns
trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]
trainML = trainML.dropna()
trainML.isnull().sum()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
X_Age = trainML[['Age']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Age,y)
# Make a prediction
y_predict = lr.predict(X_Age)
y_predict[:10]
(y == y_predict).mean()
#The prediction accuracy is marginally better than the base line accuracy of 61.5% which we got earlier
X_Fare = trainML[['Fare']].values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_Fare,y)
# Make a prediction
y_predict = lr.predict(X_Fare)
y_predict[:10]
(y == y_predict).mean()
X_sex = pd.get_dummies(trainML['Sex']).values
y = trainML['Survived'].values
# Use the fit method to train
lr.fit(X_sex, y)
# Make a prediction
y_predict = lr.predict(X_sex)
y_predict[:10]
(y == y_predict).mean()
X_pclass = pd.get_dummies(trainML['Pclass']).values
y = trainML['Survived'].values
lr = LogisticRegression()
lr.fit(X_pclass, y)
# Make a prediction
y_predict = lr.predict(X_pclass)
y_predict[:10]
(y == y_predict).mean()
from sklearn.ensemble import RandomForestClassifier
X=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Name_len', 'FamilyCount']].values # Taking all the numerical values
y = trainML['Survived'].values
RF = RandomForestClassifier()
RF.fit(X, y)
# Make a prediction
y_predict = RF.predict(X)
y_predict[:10]
(y == y_predict).mean()
test_dataset = pd.read_csv('../input/test.csv')


def data_process(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    
    
    data = data.drop(['Fare'], axis=1)
    data = data.drop(['Ticket'], axis=1)
    data = data.drop(['Cabin'], axis=1)
    freq_port = test_dataset.Embarked.dropna().mode()[0]


    data['Embarked'] = data['Embarked'].fillna(freq_port)

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    
    data = data.drop(['Name'], axis=1)
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
      
    data.loc[ data['Age'] <= 16, 'Age'] = int(0)
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age']
    
    return data
import utils
X_test_dataset = data_process(test_dataset)
X_test_dataset = X_test_dataset.drop("PassengerId", axis=1)
X_test_dataset.head()
predicted_value = RF.predict(X_test_dataset)
test_dataset_copy = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predicted_value
})

submission.to_csv('submission.csv', index=False)