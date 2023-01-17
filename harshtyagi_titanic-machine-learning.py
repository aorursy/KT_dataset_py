import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = 8, 10
X = pd.read_csv('../input/train.csv')   #TRAIN SET

Y = pd.read_csv('../input/test.csv')    #TEST SET
X.head()
Y.head()
X["Embarked"] = X["Embarked"].fillna("S")

X["Age"].fillna(Y["Age"].mean(), inplace=True)

X["Fare"].fillna(X["Fare"].mean(), inplace=True)
X.describe()
X[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
X[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
X[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x = X['Survived'], y = X['Age'], data = X)
sns.barplot(y = X['Survived'], x = X['Pclass'], data = X)

sns.barplot(y = X['Survived'], x = X['Embarked'], data = X)

sns.barplot(y = X['Pclass'], x = X['Embarked'], data = X)

sns.countplot(x=X['Survived'], hue=X['Embarked'], data=X)

# Finding out the pecentage of people on-board from each city.

filter_s = (X['Embarked'] == 'S') & (X['Survived'] == 1)

filter_q = (X['Embarked'] == 'Q') & (X['Survived'] == 1)

filter_c = (X['Embarked'] == 'C') & (X['Survived'] == 1)

len(X[filter_s])

labels = ['Southampton', 'Queenstown', 'Cherbourg']

values = [len(X[filter_s]), len(X[filter_q]), len(X[filter_c])]



trace = go.Pie(labels=labels, values=values)



py.iplot([trace], filename='basic_pie_chart')

sns.barplot(x = X['Sex'], y = X['Survived'], data = X)

heatMap = X.corr()

f, ax = plt.subplots(figsize=(25,16))

sns.plt.yticks(fontsize=18)

sns.plt.xticks(fontsize=18)



sns.heatmap(heatMap, cmap='inferno', linewidths=0.1,vmax=1.0, square=True, annot=True)
#Encoding the Sex and Embarked columns for making a heat map.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()

X.iloc[:,4] = labelencoder_x.fit_transform(X.iloc[:,4]) 



labelencoder_x2 = LabelEncoder()

X['Embarked'] = labelencoder_x2.fit_transform(X['Embarked'])

drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

X = X.drop(drop_columns, axis = 1)
X.head(3)
X.info()
#Moving the Survived column to the end by droping it and then again adding it to the data set

survived = X['Survived']

train_set = X.drop('Survived', axis = 1)  #Train set contains the whole train dataset with Survived column at the end.

train_set['Survived'] = survived
#Encoding the Sex and Embarked columns for computations.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()

train_set['Sex'] = labelencoder_x.fit_transform(train_set['Sex']) 



labelencoder_x2 = LabelEncoder()

train_set['Embarked'] = labelencoder_x2.fit_transform(train_set['Embarked'])

#Splitting the test set form train set

train_set_train = train_set.iloc[:, :-1].values

train_set_test = train_set.iloc[:, 6].values
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_set_train, train_set_test, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#Filling the missing values is Embarked Column

Y["Embarked"] = Y["Embarked"].fillna("S") 



#Storing Passenger Ids for further use

passengerID = Y['PassengerId']    



#Droping these columns as they are not required

drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']  

Y = Y.drop(drop_columns, axis = 1)



#Encoding the Sex and Embarked columns.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_Y = LabelEncoder()

Y['Sex'] = labelencoder_Y.fit_transform(Y['Sex'])



labelencoder_Y2 = LabelEncoder()

Y['Embarked'] = labelencoder_Y2.fit_transform(Y['Embarked'])

Y["Age"].fillna(Y["Age"].mean(), inplace=True)

Y["Fare"].fillna(Y["Fare"].mean(), inplace=True)
from sklearn.linear_model import LogisticRegression

logclassifier = LogisticRegression(random_state = 0)

logclassifier.fit(x_train, y_train)

y_predlg = logclassifier.predict(Y)

lcScore = logclassifier.score(x_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

KMclassifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)

KMclassifier.fit(x_train, y_train)

y_pred2 = KMclassifier.predict(Y)

KMScore = KMclassifier.score(x_train, y_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_svm = sc.fit_transform(x_train)

x_test_svm = sc.transform(x_test)



from sklearn.svm import SVC

SVMclassifier = SVC(kernel = 'poly', degree = 3, random_state = 0)

SVMclassifier.fit(x_train_svm, y_train)

SVMScore = SVMclassifier.score(x_train, y_train)
from sklearn.naive_bayes import GaussianNB

NBclassifier = GaussianNB()

NBclassifier.fit(x_train, y_train)

y_predNB = NBclassifier.predict(Y)

NBScore = SVMclassifier.score(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier

DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth=40, min_samples_split=2, min_samples_leaf=1)

DTclassifier.fit(x_train, y_train)

y_predDT = DTclassifier.predict(Y)

DTScore = DTclassifier.score(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier

RFclassifier = RandomForestClassifier(n_estimators=5, n_jobs = 2,criterion = 'entropy', random_state = 0, max_depth=14)

RFclassifier.fit(x_train, y_train)

y_predRF = RFclassifier.predict(Y)

RFTScore = RFclassifier.score(x_train, y_train)
Scores = pd.DataFrame({'Classifiers': ['Logistic', 'KMap', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest Tree'],

                      'Scores': [lcScore, KMScore, SVMScore, NBScore, DTScore, RFTScore]})
Scores
answer = pd.DataFrame({'passengerID': passengerID, 'Survived': y_pred2})

answer.head()
cols = answer.columns.tolist()

cols = cols[-1:] + cols[:-1]

answer = answer[cols]

answer.head()
answer.to_csv('answer.csv', index=False, encoding='utf-8')