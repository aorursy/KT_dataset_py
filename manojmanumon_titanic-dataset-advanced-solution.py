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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px



from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.metrics import confusion_matrix,log_loss,accuracy_score

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,StackingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler



np.random.seed(0)

sns.set_palette('pastel')
train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')

train.tail()
test.head()
train.head()
print(train.shape)

print(test.shape)
train.info()
test.info()
train.describe()
test.describe()
train.dtypes
train.isnull().sum()
test.isnull().sum()
avg_age_train = (train.groupby("Sex")['Age']).mean()

print(avg_age_train)

avg_age_test = (test.groupby("Sex")['Age']).mean()

print(avg_age_test)
for i in range(len(train['Age'])):

    if train['Age'].isnull()[i] == True and train['Sex'][i] == 'male':

        train['Age'][i] = np.round(avg_age_train['male'],decimals=1)

    elif train['Age'].isnull()[i] == True and train['Sex'][i] == 'female':

        train['Age'][i] = np.round(avg_age_train['female'],decimals=1)

        

for i in range(len(test['Age'])):

    if test['Age'].isnull()[i] == True and test['Sex'][i] == 'male':

        test['Age'][i] = np.round(avg_age_test['male'],decimals=1)

    elif test['Age'].isnull()[i] == True and test['Sex'][i] == 'female':

        test['Age'][i] = np.round(avg_age_test['female'],decimals=1)
train = train.reset_index(drop=True)

test = test.reset_index(drop=True)
print(train.shape)

print(test.shape)
train.isnull().sum()
test.isnull().sum()
Y = train['Survived']

train = train.drop('Survived',axis=1)

data = pd.concat([train,test],axis=0)

data.head()
avg_fare = data.groupby("Sex")['Fare'].mean()

avg_fare
print("Index of the null value is: ",test[test['Fare'].isnull()].index.tolist())

print(test['Sex'][152])
data['Fare'][152] = avg_fare['male']
data.isnull().sum()
print("Number of duplicate rows in the train dataset :",train.duplicated().sum())

print("Number of duplicate rows in the test dataset :",test.duplicated().sum())
Name_title_data = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(Name_title_data)

data['Name_title'] = Name_title_data

data = data.reset_index(drop=True)

data.head()
age_group_data = [None] * len(data['Age'])

for i in range(len(data['Age'])):

    if data['Age'][i] <= 3:

        age_group_data[i] = 'Baby'

    elif data['Age'][i] >3 and data['Age'][i] <= 13:

        age_group_data[i] = 'Child'

    elif data['Age'][i] >13 and data['Age'][i] <= 19:

        age_group_data[i] = 'Teenager'

    elif data['Age'][i] >19 and data['Age'][i] <= 30:

        age_group_data[i] = 'Young Adult'

    elif data['Age'][i] >30 and data['Age'][i] <= 45:

        age_group_data[i] = 'Middle Aged Adult'

    elif data['Age'][i] >45 and data['Age'][i] <65:

        age_group_data[i] = 'Adult'

    else:

        age_group_data[i] = 'Old'



data['age_group'] = age_group_data
np.unique(data['Name_title'])
data['Is_Married'] = 0

data['Is_Married'].loc[data['Name_title'] == 'Mrs'] = 1

data['FamSize'] = data['SibSp'] + data['Parch'] + 1

data['Single'] = data['FamSize'].map(lambda s: 1 if s == 1 else 0)
data.head()
np.unique(data['Ticket'])

tic = data.groupby('Ticket',sort=True,group_keys=True)

groups = list(tic.groups)

togther = [None] * len(data['Ticket'])

k=0

for i in range(len(groups)):

    for j in range(len(data['Ticket'])):

        if data['Ticket'][j] == groups[i]:

            togther[j] = i

data['Togther'] = togther
data.head()
np.unique(data['Fare'])
rates = [None]*len(data['Fare'])

for i in range(len(data['Fare'])):

    if data['Fare'][i]<=10:

        rates[i] = 1

    elif data['Fare'][i] >10 and data['Fare'][i]<=30:

        rates[i] = 2

    elif data['Fare'][i] >30 and data['Fare'][i]<=70:

        rates[i] = 3

    elif data['Fare'][i] >70 and data['Fare'][i]<=100:

        rates[i] = 4 

    else:

        rates[i] = 5

data['Rates'] = rates
data['Cabin_present'] = 1

data['Cabin_present'].loc[data['Cabin'].isnull()] = 0
data.shape
data = data.drop('Cabin',axis=1)

data = data.drop('Ticket',axis=1)

data = data.drop('Name',axis=1)

data = data.drop('PassengerId',axis=1)
data_ohe = pd.get_dummies(data,drop_first=True)

data_ohe.head()
plt.figure(figsize=(20,20))

sns.heatmap(train.corr(),annot=True,fmt="0.3f",cmap='GnBu',linewidth=1.2,linecolor='black',square=True)

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(test.corr(),annot=True,fmt="0.3f",cmap='YlOrBr',linewidth=1.2,linecolor='black',square=True)

plt.show()
train['Survived'] = Y
plt.figure(figsize=(20,12))



plt.subplot(2,2,1)

sns.countplot('Sex',data=train,palette=['darkblue','red'])

plt.title("Train data Sex Count")

plt.grid()



plt.subplot(2,2,2)

sns.countplot('Sex',data=test,palette=['teal','purple'])

plt.title("Test data Sex Count")

plt.grid()



plt.show()
plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

sns.countplot('Survived',data=train,palette=['black','yellow'],hue='Sex')

plt.grid()

plt.title("Survival Graph")



plt.subplot(2,2,2)

sns.countplot('SibSp',data=train,palette=['green','pink'],hue='Sex')

plt.grid()

plt.title("No of siblings / spouses aboard the Titanic")



plt.subplot(2,2,3)

sns.countplot('Parch',data=train,palette=['orange','greenyellow'],hue='Sex')

plt.grid()

plt.title("No of parents / children aboard the Titanic")

plt.legend(loc='upper right')



plt.subplot(2,2,4)

sns.countplot('Pclass',data=train,palette=['brown','magenta'],hue='Sex')

plt.grid()

plt.title("Passenger Class with sex")

plt.legend(loc='upper right')



plt.show()
plt.figure(figsize=(20,20))

sns.countplot(y='Age',data=train)

plt.grid()

plt.title("Train data Age ranges")

plt.show()
plt.figure(figsize=(20,20))

sns.countplot(y='Age',data=test)

plt.grid()

plt.title("Test data Age ranges")

plt.show()
plt.figure(figsize=(20,12))



plt.subplot(2,2,1)

sns.countplot('Embarked',data=train,hue='Survived',palette=['red','purple'])

plt.grid()

plt.title("Embarked plotted")



plt.subplot(2,2,2)

sns.countplot('Pclass',data=train,hue='Survived',palette=['teal','darkblue'])

plt.grid()

plt.title("Types of Passenger Classes")



plt.subplot(2,2,3)

sns.countplot('Parch',data=train,palette=['orange','greenyellow'],hue='Survived')

plt.grid()

plt.title("No of parents / children aboard the Titanic")

plt.legend(loc='upper right')



plt.subplot(2,2,4)

sns.countplot('SibSp',data=train,palette=['brown','magenta'],hue='Survived')

plt.grid()

plt.title("No of siblings / spouses aboard the Titanic")

plt.legend(loc='upper right')



plt.show()
fig = px.pie(train,names='Sex',color='Survived')

fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))

fig.show()
fig = px.pie(train,names='Embarked',color='Survived',color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))

fig.show()
fig = px.pie(train,names='Pclass',color='Survived',color_discrete_sequence=px.colors.sequential.GnBu)

fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))

fig.show()
fig = px.pie(train,names='SibSp',color='Survived',template='seaborn')

fig.update_traces(rotation=140,pull=0.01,marker=dict(line=dict(color='#000000',width=1.2)))

fig.show()
fig = px.violin(train,x='Sex',y='Age',points='all',box=True,color='Survived')

fig.show()



fig = px.violin(train,x='Sex',y='Pclass',points='all',box=True,color='Survived')

fig.show()



fig = px.violin(train,x='Sex',y='SibSp',points='all',box=True)

fig.show()
fig = px.violin(train,x='Survived',y='Age',points='all',box=True,color='Survived')

fig.show()



fig = px.violin(train,x='Survived',y='Pclass',points='all',box=True,color='Survived')

fig.show()



fig = px.violin(train,x='Survived',y='SibSp',points='all',box=True)

fig.show()
fig = px.scatter(train,x='Age',y='Fare',color='Survived',size='Age')

fig.show()



fig = px.scatter(train,x='Age',y='Fare',color='Sex',size='Age')

fig.show()
train = train.drop('Survived',axis=1)
train_ohe = data_ohe[:train.shape[0]]

test_ohe = data_ohe[train.shape[0]:]
len(data)
X_train,X_test,Y_train,Y_test = train_test_split(train_ohe,Y,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
def plot_conf_matrix(Y_test,Y_pred):

    conf = confusion_matrix(Y_test,Y_pred)

    recall =(((conf.T)/(conf.sum(axis=1))).T)

    precision =(conf/conf.sum(axis=0))



    print("Confusion Matrix : ")

    class_labels = [0,1]

    plt.figure(figsize=(10,8))

    sns.heatmap(conf,annot=True,fmt=".3f",cmap="GnBu",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("Precision Matrix ; ")

    plt.figure(figsize=(10,8))

    sns.heatmap(precision,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("Recall Matrix ; ")

    plt.figure(figsize=(10,8))

    sns.heatmap(recall,annot=True,fmt=".3f",cmap="Blues",xticklabels=class_labels,yticklabels=class_labels,linecolor='black',linewidth=1.2)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
params = dict(

    n_estimators = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],

    criterion = ['gini','entropy'],

    max_depth = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],

    min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],

)

rf = RandomForestClassifier()

clf = RandomizedSearchCV(rf,params,random_state=0,verbose=0,n_jobs=-1,n_iter=20,cv=10)

rsc = clf.fit(X_train,Y_train)

rsc.best_params_
rf = RandomForestClassifier(n_estimators=15,min_samples_leaf=6,max_depth=400,criterion='gini')

rf.fit(X_train,Y_train)

pred = rf.predict(X_test)

acc = accuracy_score(Y_test,pred)*100

print(acc)

plot_conf_matrix(Y_test,pred)
params = dict(

    learning_rate = [0.001,0.01,0.1,1,10,100,1000],

    n_estimators = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],

    criterion = ['friedman_mse','mse','mae'],

    max_depth = [2,5,10,15,20,25,30,40,50,70,100,125,150,200,300,400,500,700,1000],

    min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],

)

gbdt = GradientBoostingClassifier()

clf = RandomizedSearchCV(gbdt,params,random_state=0,verbose=0,n_jobs=-1,n_iter=20,cv=10)

gb = clf.fit(X_train,Y_train)

gb.best_params_
gbdt = GradientBoostingClassifier(n_estimators=700,min_samples_leaf=8,max_depth=1000,criterion='mse',learning_rate=0.01)

gbdt.fit(X_train,Y_train)

pred = gbdt.predict(X_test)

acc = accuracy_score(Y_test,pred)*100

print(acc)

plot_conf_matrix(Y_test,pred)
vc = VotingClassifier(estimators=[('rf', rf), ('gbdt', gbdt)],voting='soft')

vc = vc.fit(X_train,Y_train)



pred = vc.predict(X_test)

acc = accuracy_score(Y_test,pred)*100

print(acc)

plot_conf_matrix(Y_test,pred)
X_train.shape
test['PassengerId']
predictions = gbdt.predict(test_ohe)

predictions.shape
submit = pd.DataFrame(test['PassengerId'],columns=['PassengerId'])

submit['Survived'] = predictions

submit.head()
submit.to_csv("Submissions.csv",index=False)

print("Finished saving the file")