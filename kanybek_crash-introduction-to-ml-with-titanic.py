# This notebook helps to understand how to use ML tools in real world data sets. Particularly i used Titanic data 
#to help predict survived unsurvived families
# As you go down you will see following steps of my work;
#>>importing necessary libraries
#>>reading file
#>>generating and description about data contents
#>>visualization
#>>filling and replacing operations
#>>training and prediction
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KDTree, BallTree, KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
sns.set_style('whitegrid')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.head()
train.sample(n=10)
train.describe(include='all')

train.isnull().sum()
plt.hist(train['Survived'], bins = 3)
plt.show()
print("Not Survived {} out of {} passengers.".format(len(train[train['Survived'] == 0]), len(train['Survived'])))
print("Survived {} out of {} passengers.".format(len(train[train['Survived'] == 1]), len(train['Survived'])))
#the most frequent letter in Embarked feature
train['Embarked'].mode()
#replacing missing cells with S
train['Embarked'].fillna('S', inplace = True)
train.isnull().any()
train['Age'].interpolate(inplace = True)
train.isnull().any()
train['Sex'].replace('male', 1, inplace = True)
train['Sex'].replace('female', 0, inplace = True)
train['Embarked'].replace('S', 2, inplace = True)
train['Embarked'].replace('C', 1, inplace = True)
train['Embarked'].replace('Q', 0, inplace = True)
train.head()
age = train['Age'].values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
train['Age'] = pd.DataFrame(age_scaled)

fare = train['Fare'].values.reshape(-1,1)
min_max_scaler2 = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler2.fit_transform(fare)
train['Fare'] = pd.DataFrame(fare_scaled)

pclass = train['Pclass'].values.reshape(-1,1)
min_max_scaler3 = preprocessing.MinMaxScaler()
pclass_scaled = min_max_scaler3.fit_transform(pclass)
train['Pclass'] = pd.DataFrame(pclass_scaled)

emb = train['Embarked'].values.reshape(-1,1)
min_max_scaler4 = preprocessing.MinMaxScaler()
emb_scaled = min_max_scaler4.fit_transform(emb)
train['Embarked'] = pd.DataFrame(emb_scaled)

sib = train['SibSp'].values.reshape(-1,1)
min_max_scaler5 = preprocessing.MinMaxScaler()
sib_scaled = min_max_scaler5.fit_transform(sib)
train['SibSp'] = pd.DataFrame(sib_scaled)

parch = train['Parch'].values.reshape(-1,1)
min_max_scaler6 = preprocessing.MinMaxScaler()
parch_scaled = min_max_scaler6.fit_transform(parch)
train['Parch'] = pd.DataFrame(parch_scaled)


train.head()
# corr matrix
corr = train.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
sns.factorplot('Age','Survived', data=train,size=10,aspect=2)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
trainX, testX, trainY, testY = train_test_split(train[['Pclass', 'Sex', 'Age','SibSp','Parch','Fare', 'Embarked']], train['Survived'], test_size = 0.25)
#Decision tree classifier
a=np.arange(2,10,1)
for i in a:
    clf1 = DecisionTreeClassifier(max_depth=i)
    clf1.fit(trainX, trainY)
    accuracy = clf1.score(testX, testY)
    print(i,accuracy)
#Support Vector Classifier
clf2 = svm.SVC()
clf2.fit(trainX, trainY)
accuracy = clf2.score(testX, testY)
print(accuracy)
#LRC
clf3 = LogisticRegressionCV(Cs=1000, penalty='l2')
clf3.fit(trainX, trainY)
accuracy = clf3.score(testX, testY)
print(accuracy)
#KNN
clf4 = KNeighborsClassifier()
clf4.fit(trainX, trainY)
accuracy = clf4.score(testX, testY)
print(accuracy)
#Multilayer perceptron

clf5 = MLPClassifier()
clf5.fit(trainX, trainY)
accuracy = clf5.score(testX, testY)
print(accuracy)
#Random Forest Classifier
clf6 = RandomForestClassifier()
clf6.fit(trainX, trainY)
accuracy = clf6.score(testX, testY)
print(accuracy)
test.head()
test = test[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test.isnull().any() #passengerId, Name, Ticket, Cabin delete
test['Age'].interpolate(inplace = True)
test.isnull().any()

test['Embarked'].fillna('S', inplace = True)
test.isnull().any()
test['Fare'].interpolate(inplace = True)
test.isnull().any()
test['Sex'].replace('male', 1, inplace = True)
test['Sex'].replace('female', 0, inplace = True)
test['Embarked'].replace('S', 2, inplace = True)
test['Embarked'].replace('C', 1, inplace = True)
test['Embarked'].replace('Q', 0, inplace = True)
test.head()
from sklearn import preprocessing
age = test['Age'].values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
test['Age'] = pd.DataFrame(age_scaled)

fare = test['Fare'].values.reshape(-1,1)
min_max_scaler2 = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler2.fit_transform(fare)
test['Fare'] = pd.DataFrame(fare_scaled)

pclass = test['Pclass'].values.reshape(-1,1)
min_max_scaler3 = preprocessing.MinMaxScaler()
pclass_scaled = min_max_scaler3.fit_transform(pclass)
test['Pclass'] = pd.DataFrame(pclass_scaled)

emb = test['Embarked'].values.reshape(-1,1)
min_max_scaler4 = preprocessing.MinMaxScaler()
emb_scaled = min_max_scaler4.fit_transform(emb)
test['Embarked'] = pd.DataFrame(emb_scaled)
test.head()
test.head()
trainX = train[['Pclass', 'Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
trainY = train['Survived']
testX = test[['Pclass', 'Sex','Age','SibSp','Parch', 'Fare', 'Embarked']]
result = pd.DataFrame(test['PassengerId'])
clf = DecisionTreeClassifier()
clf.fit(trainX, trainY)
predictDTC = clf.predict(testX)
result['Survived'] = predictDTC

