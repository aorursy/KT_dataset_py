import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

all = [train_data,test_data]
train_data.head()
for data in all:

    data.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace=True)
train_data.info()

print('*'*40)

test_data.info()
train_data['Sex'].unique()
for data in all:

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_data.head()
for data in all:

   for i in range(0,2):

      for j in range(1,4):     

          age_mean = data[(data['Sex'] == i) & (data['Pclass'] == j)]['Age'].dropna().mean()

          data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j),'Age'] = age_mean
train_data['Embarked'] = train_data['Embarked'].fillna(train_data.Embarked.mode(dropna=True)[0])
for data in all:

    data['Embarked'] = data['Embarked'].map( {'S': 1, 'C': 2,'Q' : 3} )



train_data.head()
test_data.loc[(data.Fare.isnull())]
test_data['Fare'].fillna(test_data[(test_data['Pclass'] == 3)]['Pclass'].dropna().mean(),inplace = True)
train_data.info()

print('*'*40)

test_data.info()
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
X_train = train_data.drop(['Survived'],axis = 1)

y_train = train_data.Survived
from sklearn.preprocessing import StandardScaler

scalerModel = StandardScaler()

X_train = scalerModel.fit_transform(X_train)

test_data = scalerModel.fit_transform(test_data)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,shuffle = True,random_state=42)

train_scores = []

test_scores = []
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=42,C=1.0,max_iter=200)

classifier.fit(X_train,y_train)



#calculate the details Logistic Regression

print('train_score classifier',classifier.score(X_train,y_train))

print('test_score classifier',classifier.score(X_test,y_test))



train_scores.append(classifier.score(X_train,y_train))

test_scores.append(classifier.score(X_test,y_test))

from sklearn.svm import SVC

svcmodel = SVC(kernel='rbf',max_iter=3000,C=.10,gamma='auto',degree=3,random_state=42)

svcmodel.fit(X_train,y_train)



#calculate the details SVM

print('train_score svcmodel', svcmodel.score(X_train,y_train))

print('test_score svcmodel',svcmodel.score(X_test,y_test))



train_scores.append(svcmodel.score(X_train,y_train))

test_scores.append(svcmodel.score(X_test,y_test))



from sklearn.neighbors import KNeighborsClassifier

Knnclassifier_model = KNeighborsClassifier(n_neighbors=100 )

Knnclassifier_model.fit(X_train,y_train)



#calculate the details KNeighborsClassifier

print('train_score Knnclassifier_model', Knnclassifier_model.score(X_train,y_train))

print('test_score Knnclassifier_model',Knnclassifier_model.score(X_test,y_test))



train_scores.append(Knnclassifier_model.score(X_train,y_train))

test_scores.append(Knnclassifier_model.score(X_test,y_test))



from sklearn.naive_bayes import GaussianNB

gussian_model = GaussianNB(priors=None, var_smoothing=1e-09 )

gussian_model.fit(X_train,y_train)



#calculate the details Naive Bayes

print('train_score gussian_model', gussian_model.score(X_train,y_train))

print('test_score gussian_model',gussian_model.score(X_test,y_test))



train_scores.append(gussian_model.score(X_train,y_train))

test_scores.append(gussian_model.score(X_test,y_test))



from sklearn.tree import DecisionTreeClassifier

DT_model=DecisionTreeClassifier(criterion='entropy',random_state=42)

DT_model.fit(X_train,y_train)



#calculate the details DecisionTreeClassifier Model

print('train_score DT_model', DT_model.score(X_train,y_train))

print('test_score DT_model',DT_model.score(X_test,y_test))



train_scores.append(DT_model.score(X_train,y_train))

test_scores.append(DT_model.score(X_test,y_test))



from sklearn.neural_network import MLPClassifier



mlp_model = MLPClassifier(hidden_layer_sizes=100 ,activation='relu',alpha=0.01,epsilon=1E-08)

mlp_model.fit(X_train,y_train)



#calculate the details NNClassifier Model

print('train_score mlp_model', mlp_model.score(X_train,y_train))

print('test_score mlp_model',mlp_model.score(X_test,y_test))



train_scores.append(mlp_model.score(X_train,y_train))

test_scores.append(mlp_model.score(X_test,y_test))



from sklearn.ensemble import RandomForestClassifier



RFC_Model= RandomForestClassifier(criterion='gini',n_estimators=100,max_depth=3,random_state=42)

RFC_Model.fit(X_train,y_train)



#calculate the details RandomForestClassifier Model

print('train_score rfc', RFC_Model.score(X_train,y_train))

print('test_score rfc',RFC_Model.score(X_test,y_test))



train_scores.append(RFC_Model.score(X_train,y_train))

test_scores.append(RFC_Model.score(X_test,y_test))



Labels = ['classifier', 'svcmodel', 'Knnclassifier_model', 'gussian_model','DT_model','mlp_model','RFC_Model']

X = np.arange(1,8)

fig = plt.figure(figsize=(10,7))

ax = fig.add_axes([0,0,1,1])

plt.style.context('ggplot')

ax.bar(X + 0.00, train_scores, color = 'b', width = 0.45,label = 'train Score')

ax.bar(X + 0.40, test_scores, color = 'r', width = 0.45,label = 'test Score')

for i,m in list(zip(X,train_scores)):

  plt.text(x = i ,y = m,s = float("{:.2f}".format(m)))

for i,m in list(zip(X,test_scores)):

  plt.text(x = i + 0.45 ,y = m,s = float("{:.2f}".format(m)))

ax.set_xlabel('Models')

ax.set_ylabel('Scores')

ax.set_xticks(X)

ax.set_xticklabels(Labels)

plt.legend()
y_pred = RFC_Model.predict(test_data)
gs = pd.read_csv('../input/titanic/gender_submission.csv')

submission = pd.DataFrame({'PassengerId': gs.PassengerId, 'Survived': y_pred})

submission.to_csv('my_submission.csv', index=False)