import numpy as np 

import pandas as pd
#read the files 

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

all = [train_data,test_data]
#show the data 

train_data.head()
for data in all:

    data.drop(['PassengerId','Name','Ticket'],axis = 1,inplace=True)
train_data.info()

print('*'*40)

test_data.info()
train_data['Sex'].unique()
for data in all:

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_data.head()
train_data.info()
for data in all:

    data.drop(['Cabin'],axis = 1,inplace= True)
import seaborn as sns 

import matplotlib.pyplot as plt

sns.set_style("whitegrid")

g = sns.FacetGrid(train_data,col = 'Pclass',height = 5,row = 'Survived')

g.map(plt.hist,'Age')
female = train_data[(train_data['Sex'] == 1) & (train_data['Survived'] == 1)]['Sex'].value_counts()[1]/train_data[(train_data['Survived'] == 1)]['Survived'].value_counts()[1]

male = train_data[(train_data['Sex'] == 0) & (train_data['Survived'] == 1)]['Sex'].value_counts()[0]/train_data[(train_data['Survived'] == 1)]['Survived'].value_counts()[1]

labels = ["females Survived",'Male survived']

fig1, ax1 = plt.subplots()

ax1.pie([female,male], labels=labels,shadow=True,startangle=90,autopct='%1.1f%%')

plt.show()
g = sns.FacetGrid(train_data,col = 'Survived',height = 5)

g.map(plt.hist,'Sex')
g = sns.FacetGrid(train_data,col = 'Pclass',height = 5,row = "Sex")

g.map(plt.hist,'Age')
for data in all:

   for i in range(0,2):

      for j in range(1,4):     

          age_mean = data[(data['Sex'] == i) & (data['Pclass'] == j)]['Age'].dropna().mean()

          data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j),'Age'] = age_mean
train_data.info()

print('*'*40)

test_data.info()
train_data['Embarked'] = train_data['Embarked'].fillna(train_data.Embarked.mode(dropna=True)[0])
test_data.loc[(data.Fare.isnull())]
test_data['Fare'].fillna(test_data[(test_data['Pclass'] == 3)]['Pclass'].dropna().mean(),inplace = True)

for data in all:

    data['Family'] = data['Parch'] + data['SibSp'] + 1

    data.drop(['Parch','SibSp'],axis = 1,inplace = True)

train_data.info()

print('*'*40)

test_data.info()
for data in all:

    data['Embarked'] = data['Embarked'].map( {'S': 1, 'C': 2,'Q' : 3} )



train_data.head()
X_train = train_data.drop(['Survived'],axis = 1)

y_train = train_data.Survived
from sklearn.preprocessing import StandardScaler

scalerModel = StandardScaler()

X_train = scalerModel.fit_transform(X_train)

test_data = scalerModel.fit_transform(test_data)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.21,shuffle = True,random_state=33)
train_scores = []

test_scores = []
from sklearn.svm import SVC





SVCModel = SVC(kernel= 'rbf',

               max_iter=3000,C=.10,gamma='auto')

SVCModel.fit(X_train, y_train)







print('train data score',SVCModel.score(X_train,y_train))

print('test data score',SVCModel.score(X_test,y_test))

train_scores.append(SVCModel.score(X_train,y_train))

test_scores.append(SVCModel.score(X_test,y_test))
from sklearn.ensemble import RandomForestClassifier



RandomForestClassifierModel = RandomForestClassifier(criterion = 'entropy',n_estimators=300,max_depth=5,random_state=33,bootstrap=False,min_samples_leaf=3) 

RandomForestClassifierModel.fit(X_train, y_train)



#Calculating Details

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

train_scores.append(RandomForestClassifierModel.score(X_train, y_train))

test_scores.append(RandomForestClassifierModel.score(X_test, y_test))

#print('----------------------------------------------------')

from sklearn.linear_model import LogisticRegression

LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=0.5,random_state=33)

LogisticRegressionModel.fit(X_train, y_train)



#Calculating Details

print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))

print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))

print('LogisticRegressionModel No. of iteratios is : ' , LogisticRegressionModel.n_iter_)

train_scores.append(LogisticRegressionModel.score(X_train, y_train))

test_scores.append(LogisticRegressionModel.score(X_test, y_test))

print('----------------------------------------------------')

from sklearn.tree import DecisionTreeClassifier



DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=33) 

DecisionTreeClassifierModel.fit(X_train, y_train)



#Calculating Details

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))

print('----------------------------------------------------')

train_scores.append(DecisionTreeClassifierModel.score(X_train, y_train))

test_scores.append(DecisionTreeClassifierModel.score(X_test, y_test))
Labels = ['SVC', 'RFC', 'LR', 'DTC']

X = np.arange(1,5)

fig = plt.figure(figsize=(10,7))

ax = fig.add_axes([0,0,1,1])

plt.style.context('ggplot')

ax.bar(X + 0.00, train_scores, color = 'b', width = 0.40,label = 'train accuracy')

ax.bar(X + 0.40, test_scores, color = 'r', width = 0.40,label = 'test accuracy')

for i,m in list(zip(X,train_scores)):

  plt.text(x = i ,y = m,s = float("{:.2f}".format(m)))

for i,m in list(zip(X,test_scores)):

  plt.text(x = i + 0.45 ,y = m,s = float("{:.2f}".format(m)))

ax.set_xlabel('Models')

ax.set_ylabel('accuracy')

ax.set_xticks(X)

ax.set_xticklabels(Labels)

plt.legend()
y_pred = RandomForestClassifier.predict(test_data)
gs = pd.read_csv('../input/titanic/gender_submission.csv')

submission = pd.DataFrame({'PassengerId': gs.PassengerId, 'Survived': y_pred})

submission.to_csv('my_submission.csv', index=False)