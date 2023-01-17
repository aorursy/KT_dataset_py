# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
original_train_data = pd.read_csv('../input/titanic/train.csv')

original_test_data = pd.read_csv('../input/titanic/test.csv')

original_gender = pd.read_csv('../input/titanic/gender_submission.csv')
train_data = original_train_data.copy()

test_data = original_test_data.copy()
test_data.head()
train_data.head(10)
train_data.info()
train_data.describe()
train_data.head(15)
total = train_data.isnull().sum().sort_values(ascending = False)

percent_1 = train_data.isnull().sum() / train_data.isnull().count() * 100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis = 1, keys = ['Total','%'])

missing_data.head(10)
train_data.columns.values #train_data.columns
import seaborn as sns

survived = "Survived"

not_survived = "Not Survived"

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

women = train_data[train_data['Sex'] == 'female']

men = train_data[train_data['Sex'] == 'male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde =False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde =False)

ax.legend()

ax.set_title('Male')
## Lets See in separate Graph

survived = "Survived"

not_survived = "Not Survived"

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 6))

women = train_data[train_data['Sex'] == 'female']

men = train_data[train_data['Sex'] == 'male']

ax1 = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived,color = 'green', ax = axes[0][0], kde =False)

ax2 = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived,color = 'red', ax = axes[0][1], kde =False)

ax1.legend()

ax1.set_title('Survival Female')

ax2.legend()

ax2.set_title('Not Survival Female')



ax1 = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived,color = 'green', ax = axes[1][0], kde =False)

a2 = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = not_survived,color = 'red', ax = axes[1][1], kde =False)

ax1.legend()

ax1.set_title('Survival Male')

ax2.legend()

ax2.set_title('Not Survival Male')

fig. tight_layout(pad=3.0)

train_data.head(5)
pclass = train_data.groupby(['Pclass','Survived'])['Survived'].count()

pclass
pclass1 = train_data.groupby(['Pclass','Survived'])['Survived'].count().groupby('Survived').sum()

pclass1
datag = pd.DataFrame(pclass)

datag
datag['Survived'][1].values
## PrePare Pie char to understand it more closely

fig, axs = plt.subplots(nrows = 1, ncols = 3,figsize = (15,8))



axs[0].pie(datag['Survived'][1].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%', colors = ['red','green'])

axs[0].set_title('PClass-1 Survival Rate')



axs[1].pie(datag['Survived'][2].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%',colors = ['red','green'])

axs[1].set_title('PClass-2 Survival Rate')



axs[2].pie(datag['Survived'][3].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%',colors = ['red','green'])

axs[2].set_title('PClass-3 Survival Rate')

plt.show()



sns.barplot(x='Pclass', y='Survived', data=train_data)
train_data.head(10)
train_data.groupby('Embarked')['Embarked'].count()
embark = train_data.groupby(['Embarked','Survived'])['Survived'].count()

embark
embarkSR = pd.DataFrame(embark)

embarkSR
embarkSR['Survived']['C'].values
## PrePare Pie char to understand it more closely

fig, axs = plt.subplots(nrows = 1, ncols = 3,figsize = (15,8))

explode = (0, 0.1)

axs[0].pie(embarkSR['Survived']['C'].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%', colors = ['red','green'],shadow=True, startangle=90,explode = explode)

axs[0].set_title('Embarked-C: Survival Rate')

axs[0].axis('equal')

axs[1].pie(embarkSR['Survived']['Q'].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%',colors = ['red','green'],shadow=True, startangle=90,explode = explode)

axs[1].set_title('Embarked-Q: Survival Rate')

axs[1].axis('equal')

axs[2].pie(embarkSR['Survived']['S'].values, labels = ['Fail to Survive','Survive'],autopct='%1.2f%%',colors = ['red','green'],shadow=True, startangle=90)

axs[2].set_title('Embarked-S: Survival Rate')

axs[2].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

sns.barplot(x='Embarked', y='Survived', data=train_data)
train_data['relatives'] = train_data['SibSp'] + train_data['Parch']



train_data.loc[train_data['relatives'] > 0,'not_alone'] = 1

train_data.loc[train_data['relatives'] == 0,'not_alone'] = 0

train_data['not_alone'].value_counts()
xes = sns.factorplot('relatives','Survived', 

                      data=train_data, aspect = 2.5, )
train_data.drop(['PassengerId','Name'],axis = 1, inplace = True)

train_data.head()
cabinValues = [1,2,3,4,5,6,7,8]

Deck = train_data['Cabin'].dropna(axis = 0).str[0:1].unique()

cabinKey = np.sort(Deck, kind = 'quick')

deckPair = {cabinKey[i]: cabinValues[i] for i in range(len(cabinKey))} 



train_data['Cabin'].fillna('0',inplace = True)
def setDeck(x):

    char1 = x[0:1]

    if char1 == '0':

        return 0

    else:

        return deckPair[char1]
train_data['Deck'] = train_data['Cabin'].apply(lambda x : setDeck(x))
train_data.head(10)
train_data.drop(['Cabin'], inplace = True, axis = 1)

train_data['Deck'] = train_data['Deck'].astype('int32')
# Now deal with Age

train_data['Age'].isnull().sum()

xTrain = train_data.copy()

mean = xTrain['Age'].mean()

std = xTrain['Age'].std()

is_null = xTrain["Age"].isnull().sum()

print("Mean : {mean},Std : {std}, isNull : {isNull}".format(mean = mean, std = std,isNull = is_null))
rand_age = np.random.randint(mean - std, mean + std, size = is_null)

nullIndex = xTrain['Age'].index[xTrain['Age'].apply(np.isnan)]

sr1 = pd.Series(rand_age, index = nullIndex) # It is necessary to set indexof NaN to series,so that it can set values according to index

xTrain['Age'].fillna(value = sr1,axis = 0,inplace = True) ## fillNa with with same length and on same index

xTrain['Age'] = xTrain['Age'].astype('int32')

xTrain["Age"].isnull().sum()
xTrain['Embarked'].describe()
xTrain['Embarked'].fillna(value = 'S',axis = 0,inplace = True)
xTrain["Embarked"].isnull().sum()
xTrain.groupby('Embarked')['Embarked'].count()
ports = {"S": 0, "C": 1, "Q": 2}

xTrain['Embarked'] = xTrain['Embarked'].map(ports)

xTrain.head(10)
xTrain.info()
genders = {"male": 0, "female": 1}

xTrain['Sex'] = xTrain['Sex'].map(genders)
xTrain.head(10)
xTrain['Ticket'].describe()
xTrain.drop(['Ticket'],axis = 1, inplace = True)
xTrain.drop(['SibSp','Parch','not_alone'], axis = 1, inplace = True)

cleanedData = xTrain.copy()

cleanedData.head(10)
test_data.isnull().sum()
#test_data.drop(['PassengerId','Name'], axis = 1, inplace = True)

testData = test_data.copy()

testData['relatives'] = testData['SibSp'] + test_data['Parch']

cabinValues = [1,2,3,4,5,6,7,8]

Deck = testData['Cabin'].dropna(axis = 0).str[0:1].unique()

cabinKey = np.sort(Deck, kind = 'quick')

deckPair = {cabinKey[i]: cabinValues[i] for i in range(len(cabinKey))} 





testData['Cabin'].fillna('0',inplace = True)

testData['Deck'] = testData['Cabin'].apply(lambda x : setDeck(x))



# Now deal with Age

#testData['Age'].isnull().sum()

mean = testData['Age'].mean()

std = testData['Age'].std()

is_null = testData["Age"].isnull().sum()

print("Mean : {mean},Std : {std}, isNull : {isNull}".format(mean = mean, std = std,isNull = is_null))



rand_age = np.random.randint(mean - std, mean + std, size = is_null)

nullIndex = testData['Age'].index[testData['Age'].apply(np.isnan)]

sr1 = pd.Series(rand_age, index = nullIndex) 

testData['Age'].fillna(value = sr1,axis = 0,inplace = True) ## 



testData.dropna(axis = 0, inplace = True)
testData.head(10)
testData.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis = 1, inplace = True)

testData['Embarked'] = testData['Embarked'].map(ports)

testData['Sex'] = testData['Sex'].map(genders)
testData.head(10)
X_train = cleanedData.drop("Survived", axis=1)

Y_train = cleanedData['Survived']

x_test = testData

X_train.head()
## We Should make column on same scale

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

x_test = sc.fit_transform(x_test)
# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



scoreList =[]

alogoName =[]
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(x_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

scoreList.append(acc_log)

alogoName.append('Logistic regression')

print(round(acc_log,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(x_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



alogoName.append('Random Forest')

scoreList.append(acc_random_forest)

print(round(acc_random_forest,2,), "%")
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)



Y_pred = knn.predict(x_test)



acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

alogoName.append('K-Nearest Neighbors')

scoreList.append(acc_knn)

print(round(acc_knn,2,), "%")
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)



Y_pred = gaussian.predict(x_test)



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



alogoName.append('Gaussian')

scoreList.append(acc_gaussian)

print(round(acc_gaussian,2,), "%")
## Perceptron
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(x_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

alogoName.append('Perceptron')

scoreList.append(acc_perceptron)

print(round(acc_perceptron,2,), "%")
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

alogoName.append('Decision Tree')

scoreList.append(acc_decision_tree)

print(round(acc_decision_tree,2,), "%")
from sklearn.svm import SVC

svc = SVC(gamma='auto',kernel='rbf')

svc.fit(X_train, Y_train)



Y_pred = svc.predict(x_test)

acc_svc = round(decision_tree.score(X_train, Y_train) * 100, 2)

alogoName.append('SVC')

scoreList.append(acc_svc)

print(round(acc_svc,2,), "%")
scoreData = pd.DataFrame({

    'Algorithm' : alogoName,

    'Score' : scoreList,

    

})

#scoreData['Score'].sort_values(ascending = False)

scoreData = scoreData.sort_values(by='Score', ascending=False)

scoreData = scoreData.set_index('Score')
scoreData.head(10)
from sklearn.model_selection import KFold

def KFoldEvaluation(X, y):

    scores = []

    algoName = []

    evaluate = {'RMD' : [],'DT' : [],'KNN' : [],'LR' : [],'Gaussian' : [],'Perceptron' : [],'LR' : [],'SVC' : []}

    cv = KFold(n_splits=30, random_state=42, shuffle=False)

    for train_index, test_index in cv.split(X):

        #print("Train Index: ", train_index, "\n")

        #print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

       

        decision_tree.fit(X_train, y_train)

        evaluate.get('DT').append(decision_tree.score(X_test, y_test))

        

        perceptron.fit(X_train, y_train)

        evaluate.get('Perceptron').append(perceptron.score(X_test, y_test))

        

        gaussian.fit(X_train, y_train)

        evaluate.get('Gaussian').append(gaussian.score(X_test, y_test))

        

        knn.fit(X_train, y_train)

        evaluate.get('KNN').append(knn.score(X_test, y_test))

        

        random_forest.fit(X_train, y_train)

        evaluate.get('RMD').append(random_forest.score(X_test, y_test))

        

        logreg.fit(X_train, y_train)

        evaluate.get('LR').append(logreg.score(X_test, y_test))

        

        svc.fit(X_train, y_train)

        evaluate.get('SVC').append(svc.score(X_test, y_test))

        

    dfg = pd.DataFrame(evaluate)

    return dfg.mean(axis = 0).sort_values(ascending = False)

performance = KFoldEvaluation(X_train,Y_train)

performance
perFormanceData = pd.DataFrame(

{

    'Name' : performance.index.values,

    'Score' : performance[:].values

}

)



plt.title("KFold Performnace Graph")

sns.barplot(x='Name', y='Score', data=perFormanceData)