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
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as offline
offline.init_notebook_mode()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
train = train.fillna(1)
test = test.fillna(1)
train.info()
corr = train.corr()
corr['Survived']
train = train.drop(columns=['PassengerId', 'Name','Ticket','Cabin'])    
passengers_test = test['PassengerId']
test = test.drop(columns=['PassengerId', 'Name','Ticket','Cabin'])   
all_data = [train, test]
passenger_category_survived = train[train['Survived']==1].groupby(by = ['Pclass'])['Survived'].count().reset_index()

passenger_category_notsurvived = train[train['Survived']==0].groupby(by = ['Pclass'])['Survived'].count().reset_index()

trace6 = go.Bar(
    x=passenger_category_survived['Pclass'],
    y=passenger_category_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=passenger_category_notsurvived['Pclass'],
    y=passenger_category_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by PClass',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
for data in all_data:
    data['Sex'] = data['Sex'].mask(data['Sex'] == 'male', 1)
    data['Sex'] = data['Sex'].mask(data['Sex'] == 'female', 2)
data['Sex'].head()
sex_survived = train[train['Survived']==1].groupby(by = ['Sex'])['Survived'].count().reset_index()
sex_notsurvived = train[train['Survived']==0].groupby(by = ['Sex'])['Survived'].count().reset_index()
trace6 = go.Bar(
    x=sex_survived['Sex'],
    y=sex_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=sex_notsurvived['Sex'],
    y=sex_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by Sex',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
for data in all_data:
    data['Age'] = data['Age'].mask(data['Age'].between(0,15), 1)
    data['Age'] = data['Age'].mask(data['Age'].between(15.1,30), 2)
    data['Age'] = data['Age'].mask(data['Age'].between(30.1,45), 3)
    data['Age'] = data['Age'].mask(data['Age'].between(45.1,60), 4)
    data['Age'] = data['Age'].mask(data['Age'].between(60.1,80), 5)
data['Age'] = data['Age'].apply(int)
data['Age'].head()
age_survived = train[train['Survived']==1].groupby(by = ['Age'])['Survived'].count().reset_index()
age_notsurvived = train[train['Survived']==0].groupby(by = ['Age'])['Survived'].count().reset_index()
trace6 = go.Bar(
    x=age_survived['Age'],
    y=age_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=age_notsurvived['Age'],
    y=age_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by Age',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
for data in all_data:
     data['Family'] = data['SibSp'] + data['Parch']
# now that we have constructed a new column out of SibSp and Parch, we can drop them both
train = train.drop(columns=['SibSp','Parch'])
test = test.drop(columns=['SibSp','Parch'])
all_data = [train, test] # refresh the new data

data['Family'].head()
family_survived = train[train['Survived']==1].groupby(by = ['Family'])['Survived'].count().reset_index()
family_notsurvived = train[train['Survived']==0].groupby(by = ['Family'])['Survived'].count().reset_index()
trace6 = go.Bar(
    x=family_survived['Family'],
    y=family_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=family_notsurvived['Family'],
    y=family_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by Size of Family',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
for data in all_data:
    data['Fare'] = data['Fare'].mask(data['Fare'].between(0,7.91), 1)
    data['Fare'] = data['Fare'].mask(data['Fare'].between(7.9101,14.454), 2)
    data['Fare'] = data['Fare'].mask(data['Fare'].between(14.4541,31), 3)
    data['Fare'] = data['Fare'].mask(data['Fare'].between(31.0001,512.330), 4)
data['Fare'] = data['Fare'].apply(int)
data['Fare'].head()
fare_survived = train[train['Survived']==1].groupby(by = ['Fare'])['Survived'].count().reset_index()
fare_notsurvived = train[train['Survived']==0].groupby(by = ['Fare'])['Survived'].count().reset_index()
trace6 = go.Bar(
    x=fare_survived['Fare'],
    y=fare_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=fare_notsurvived['Fare'],
    y=fare_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by Fare',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
for data in all_data:
    data['Embarked'] = data['Embarked'].mask(data['Embarked']=='S', 1)
    data['Embarked'] = data['Embarked'].mask(data['Embarked']=='C', 2)
    data['Embarked'] = data['Embarked'].mask(data['Embarked']=='Q', 3)
    data['Embarked'] = data['Embarked'].mask(data['Embarked']==1, 1)
data['Embarked'].head()
embarked_survived = train[train['Survived']==1].groupby(by = ['Embarked'])['Survived'].count().reset_index()
embarked_notsurvived = train[train['Survived']==0].groupby(by = ['Embarked'])['Survived'].count().reset_index()
trace6 = go.Bar(
    x=embarked_survived['Embarked'],
    y=embarked_survived['Survived'],
    name='Survived'
)
trace7 = go.Bar(
    x=embarked_notsurvived['Embarked'],
    y=embarked_notsurvived['Survived'],
    name=' Not Survived'
)
    
grph = [trace6, trace7]
layout = go.Layout(
    title = 'Survival by place of Embarkment',
    barmode='stack'
)
offline.iplot({'data': grph, 
               'layout': layout})
train.head()
test.head()
train_X = train.drop(columns=['Survived'] )
train_Y = train['Survived']
test_X = test
test_Y = pd.DataFrame()
svc = SVC()
svc.fit(train_X, train_Y)
test_Y['SVM'] = svc.predict(test_X)
svc.score(train_X, train_Y)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_Y)
test_Y['KNN'] = knn.predict(test_X)
knn.score(train_X, train_Y)
gaussian = GaussianNB()
gaussian.fit(train_X, train_Y)
test_Y['Gaussian'] = gaussian.predict(test_X)
gaussian.score(train_X, train_Y)
linear_svc = LinearSVC()
linear_svc.fit(train_X, train_Y)
test_Y['linear SVM'] = linear_svc.predict(test_X)
linear_svc.score(train_X, train_Y)
decision_tree = DecisionTreeClassifier(max_depth=15, criterion='entropy')
decision_tree.fit(train_X, train_Y)
test_Y['DT'] = decision_tree.predict(test_X)
decision_tree.score(train_X, train_Y)
random_forest = RandomForestClassifier(max_depth=15, random_state=0, criterion='entropy', n_estimators=100)
random_forest.fit(train_X, train_Y)
test_Y['RF'] = random_forest.predict(test_X)
random_forest.score(train_X, train_Y)
test_Y.head(10)
submission = pd.DataFrame({
        "PassengerId": passengers_test,
        "Survived": test_Y['RF']
    })
#submission.to_csv('submission.csv', index=False)