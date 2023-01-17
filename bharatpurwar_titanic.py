# Took help from here 

# https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm , skew

from scipy import stats

import sklearn.preprocessing as StandardScaler

%matplotlib inline



#loading data

train = pd.read_csv(r"/kaggle/input/titanic/train.csv")

test = pd.read_csv(r"/kaggle/input/titanic/test.csv")



#train.head()

train.shape

#(891,12)
train.describe()
train.info()
test.info()
"""

percentage of null values to total no.of rows

"""

total_miss = train.isnull().sum()

total_miss.fillna(0)

percent_miss = train.isnull().sum()/train.shape[0] 

data_miss = pd.concat([total_miss, percent_miss], axis= 1)

data_miss = data_miss[data_miss[0] > 0]

data_miss.sort_values(by = [0],inplace = True)

data_miss.columns = ['Count','Proportion']

data_miss.index.names = ['Names']

data_miss['Names'] = data_miss.index

data_miss
train[['Pclass','Survived']].groupby('Pclass' , as_index = False).mean().sort_values(by = 'Survived' , ascending = False)
train[['Sex','Survived']].groupby('Sex' , as_index = False).mean().sort_values(by = 'Survived' , ascending = False)
train[['Embarked','Survived']].groupby('Embarked' , as_index = False).mean().sort_values(by = 'Survived' , ascending = False)
train[['SibSp','Survived']].groupby(['SibSp'] , as_index = False).mean().sort_values(by = 'Survived' , ascending = False)
train[['Parch','Survived']].groupby(['Parch'] , as_index = False).mean().sort_values(by = 'Survived' , ascending = False)
#between name and survival  

train['Title'] = train.Name.str.extract('([A-Za-z]+)\.',expand = False)

test['Title'] = test.Name.str.extract('([A-Za-z]+)\.',expand = False)



pd.crosstab(train['Title'], train['Sex'])
#analysing by visualizing data(age and survival)

g = sns.FacetGrid(train , col = 'Survived')

g.map(plt.hist , 'Age' , bins = 20)
#analysing by visualizing data(age and survival)

x1 = train.loc[train.Survived == 0, 'Age']

x2 = train.loc[train.Survived == 1, 'Age']



kwargs = dict(alpha = 0.2, bins=20)



plt.hist(x1.dropna(), **kwargs, color='g', label='Dead')

plt.hist(x2.dropna(), **kwargs, color='r', label='Survived')

plt.gca().set(title='Survival based on Age', ylabel='Total')

plt.xlim(0,100)

plt.legend();
#analysing by visualizing data(age and survival)

g = sns.FacetGrid(train , col = 'Survived', row = 'Pclass')

g.map(plt.hist , 'Age' , bins = 20)

#max pop in age bet 20 to 40 


#analysing by visualizing data(age , embarked , sex , pclass and survival)

grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
#  above result may be because male population may be much larger on "C"

# lets check

print(len(train[train.Embarked == 'S']['Sex']), " male pop = " ,len(train[train.Embarked == 'S']['Sex'][train[train.Embarked == 'S']['Sex'] == 'male'])) 

print(len(train[train.Embarked == 'C']['Sex']), " male pop = " ,len(train[train.Embarked == 'C']['Sex'][train[train.Embarked == 'C']['Sex'] == 'male'])) #so that's not the case

print(len(train[train.Embarked == 'Q']['Sex']), " male pop = " ,len(train[train.Embarked == 'Q']['Sex'][train[train.Embarked == 'Q']['Sex'] == 'male'])) 

#analysing by visualizing data( embarked , sex , fare and survival)

# grid = sns.FacetGrid(train, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()

#female passenger paid higher fares somehow...nothing useful
#combining datasets.

all_data = train.append(test)
all_data.shape
#making a new variable having total family members
all_data['total_relatives'] = all_data['SibSp'] + all_data['Parch'] 

all_data.loc[all_data['total_relatives']==0 ,'alone'] = 1
all_data.loc[all_data['total_relatives']>0 ,'alone'] = 0
all_data[['Survived','alone']].groupby(['alone']).count().sort_values(by = 'Survived')
#plotting this new variable

g = sns.FacetGrid(all_data, col = 'Survived')

g.map(plt.hist, 'total_relatives', bins = 10)

g.add_legend()
# removing passengerId

all_data.drop('PassengerId', inplace = True, axis = 1)
#cabin data

dictionary_deck = {'A':1, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'U':0}

#dictionary_deck = {'A':1, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

all_data['Cabin'].fillna('U', inplace = True)

all_data['Deck'] = all_data['Cabin'].str.extract('([a-zA-Z])', expand = True)

all_data['Deck'] = all_data['Deck'].map(dictionary_deck)
#dropping cabin column

all_data.drop('Cabin', inplace = True, axis = 1)
all_data.columns
# finding age with Sex and Pclass

age_ = sns.FacetGrid(all_data, col = 'Sex', row = 'Pclass')

age_.map(plt.hist, 'Age')

age_.add_legend()
for i in ['male', 'female']:    #Sex

    for j in [1, 2, 3]:   #Pclass 

        mean = all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j)]['Age'].mean()

        st_d = all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j)]['Age'].std()

        is_null = all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j)]['Age'].isnull().sum()

        rand_age = np.random.randint(mean - st_d, mean + st_d, is_null)

        age_slice = all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j)]['Age'].copy()

        age_slice[np.isnan(age_slice)] = rand_age

        all_data.loc[((all_data['Sex'] == i) & (all_data['Pclass'] == j)), 'Age'] = age_slice

        #print(all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j)]['Age'])
all_data['Age'].isnull().sum()
#all_data['last_name'] = all_data['Name'].str.extract('([A-Za-z]+)\,' , expand = True)
#all_data[all_data['last_name']=='Sage']

# this shows all Sage are indeed related , so if we can find one deck , we will know all others as well
#all_data['Ticket_Deck'] = all_data['Ticket'].str.extract('([a-zA-Z])', expand = True)

#dictionary_deck = {'A':1, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

#all_data['Cabin'].fillna('U', inplace = True)

#all_data['Deck'] = all_data['Cabin'].str.extract('([a-zA-Z])', expand = True)

#all_data['Ticket_Deck'] = all_data['Ticket_Deck'].map(dictionary_deck)
#all_data.drop(['Ticket_Deck'], axis = 1, inplace = True)
#all_data['last_name'].value_counts().head()
# making age bands

all_data['AgeBand'] = pd.cut(all_data['Age'], 5).cat.codes
all_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean()
all_data.drop(['Age'], axis = 1, inplace = True)
#drop parch , sibsb , total_relatives

all_data.drop(['Parch', 'SibSp', 'total_relatives'], axis = 1, inplace = True)
#drop Ticket , Name

all_data.drop(['Name', 'Ticket'], axis = 1, inplace = True)
#extracting titles from Name

#all_data['Title'] = all_data.Name.str.extract('([A-Za-z]+)\.',expand = False)

#making bands of Fare

all_data['Fare'].fillna(all_data['Fare'].median() , inplace = True)

all_data['FareBand'] = pd.qcut(all_data['Fare'] , 5).cat.codes
#dropping Fare

all_data.drop(['Fare'], axis = 1, inplace = True)
#filling null values

all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace = True)

all_data['Deck'].fillna(all_data['Deck'].mode()[0], inplace = True)
# Title 


all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')

all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')

all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

all_data['Title'] = all_data['Title'].map(title_mapping)
all_data.head()
#labeling the data

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

  

all_data['Embarked']= le.fit_transform(all_data['Embarked'])

all_data['Sex']= le.fit_transform(all_data['Sex'])

all_data['alone']= le.fit_transform(all_data['alone']) 

all_data['Deck']= le.fit_transform(all_data['Deck']) 



#Ordinal data columns are Pclass, AgeBand, FareBand

#Nominal data col are Embarked, Sex, Title, Alone, Deck



#from sklearn.preprocessing import OneHotEncoder

#encoder = OneHotEncoder(categorical_features = [0,2,4,5,6] , drop = [0])

#all_data = encoder.fit_transform(all_data)

#all_data





def encode_and_bind(original_dataframe, feature_to_encode):

    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]].astype(str))

    dummies.drop(dummies.columns[0], axis = 1, inplace = True)

    res = pd.concat([original_dataframe, dummies], axis=1)

    res = res.drop([feature_to_encode], axis=1)

    return(res)



all_data = encode_and_bind(all_data, 'Deck')

all_data = encode_and_bind(all_data, 'Embarked')
all_data.head()
X_train = all_data[~(all_data.Survived.isnull())].drop('Survived', axis = 1) 

Y_train = all_data[~(all_data.Survived.isnull())]['Survived']

X_test = all_data[all_data.Survived.isnull()].drop('Survived', axis = 1) 
X_train.shape
Y_train.shape
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# KNN

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train) 

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train) 

Y_pred = decision_tree.predict(X_test) 

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, oob_score = True)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
## Feature Importance
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
print("oob score:", round(rf.oob_score, 4)*100, "%")
"""

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}

from sklearn.model_selection import GridSearchCV, cross_val_score

rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)

clf.fit(X_train, Y_train)

clf.bestparams

"""
# Random Forest

random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)



random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
#submissions

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_prediction

    })

    

submission.to_csv(r'submission.csv', index=False) 