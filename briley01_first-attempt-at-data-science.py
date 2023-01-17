import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import bisect

import seaborn as sns



import matplotlib

matplotlib.style.use('ggplot')

%matplotlib inline
data = pd.read_csv('../input/train.csv', index_col=0)
data.head()
data.describe()
data.isnull().sum()
data[['Survived', 'Age']].corr()
titles = ['Mr\.', 'Miss\.', 'Master\.', 'Dr\.', 'Mrs\.', 'Ms\.']

data[data['Age'].isnull() & (data['Name'].str.contains('|'.join(titles)) == False)]
def NameToTitle(name):

    if 'Mr.' in name:

        return 'Mr.'

    elif 'Miss.' in name:

        return 'Miss.'

    elif 'Mrs.' in name:

        return 'Mrs.'

    elif 'Master.' in name:

        return 'Master.'

    elif 'Dr.' in name:

        return 'Dr.'

    elif 'Ms.' in name:

        return 'Ms.'

    

        

data['Title'] = data.apply(lambda row: NameToTitle(row['Name']), axis=1)
abt = data[['Age', 'Title']].groupby('Title').mean()

abt
#update ages with aveerage age for title

data.update(data[data['Age'].isnull()].apply(lambda row: abt.loc[row['Title']], axis=1))
data[['Survived', 'Age']].corr()
data['AgeGroup'] = data.apply(lambda row: bisect.bisect_left([0,1,15,65], row.Age) - 1, axis=1)

data[['Survived', 'AgeGroup']].corr()
data.drop('Name', axis=1, inplace=True)

data.drop('Title', axis=1, inplace=True)

data.drop('Age', axis=1, inplace=True)
data['Sex'] = data.apply(lambda row: 0 if(row.Sex == 'male') else 1, axis=1)
data[['Fare']].describe()
data[['Survived', 'Fare']].corr()
data['FareGroup'] = data.apply(lambda row: bisect.bisect_left([0,7,10,11,15,49,52,71,80], row.Fare) - 1, axis=1)

data[['Survived', 'FareGroup']].corr()
data.drop('Fare', axis=1, inplace=True)
data.head()
#there is one column that the ticket "number" is the word "LINE".  replace that with -1 so we can convert to ints

data['TicketNum'] = pd.to_numeric(data.Ticket.str.split(' ').str[-1].replace('LINE', -1))
data['TicketNum'].describe()
ds_tmp = data.join( data.groupby(['TicketNum'])['Cabin'].first(), on='TicketNum', rsuffix='_r')

data['Cabin'] = ds_tmp['Cabin_r']

ds_tmp['Cabin'].isnull().sum() - ds_tmp['Cabin_r'].isnull().sum()
data.boxplot(['TicketNum'])
data[['Survived', 'TicketNum']].corr()
data['TicketGroup'] = data.apply(lambda row: bisect.bisect_left([0,40000], row.TicketNum) - 1, axis=1)

data[['Survived', 'TicketGroup']].corr()
data[['TicketGroup', 'TicketNum', 'Survived']].head()
data['TicketLen'] = data['TicketNum'].astype(str).str.len()

data.boxplot(['TicketLen'])
data['TicketLenGrp'] = data.apply(lambda row: bisect.bisect_left([5,6], row.TicketLen) - 1, axis=1)

data[['Survived', 'TicketLenGrp']].corr()
data.drop('TicketLenGrp', axis=1, inplace=True)

data.drop('TicketLen', axis=1, inplace=True)

data.drop('TicketNum', axis=1, inplace=True)
data.fillna('NULL')[['Embarked', 'Survived']].groupby('Embarked').count()
mapping = {'C': 1, 'Q': 2, 'S': 3}

data['Embarked'] = data[['Embarked']].fillna('S').replace({'Embarked': mapping})
data.drop('Cabin', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)
data.isnull().sum()
corr = data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.barplot(x="FareGroup", y="Survived", hue="Pclass", data=data)
sns.barplot(x="AgeGroup", y="Survived", hue="Sex", data=data)
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'AgeGroup', 'FareGroup', 'TicketGroup']

y = data['Survived']

X = data[features]
#from https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, NuSVC

from sklearn.ensemble import RandomForestClassifier



# prepare configuration for cross validation test harness

seed = 7

# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('NuSVM', NuSVC()))

models.append(('RFC', RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score

#Choose the classifier

clf = SVC()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Choose some parameter combinations to try

param_C = [0.001, 0.01, 0.1, 1, 10]

param_gamma = [0.001, 0.01, 0.1, 1]

parameters = [{'kernel': ['rbf'], 'C': param_C, 'gamma': param_gamma}#,

              #{'kernel': ['poly'], 'degree': [2,3,4], 'C': param_C, 'gamma': param_gamma}

             ]



kfold = model_selection.KFold(n_splits=10, random_state=seed)

clf = GridSearchCV(estimator=clf, param_grid=parameters, cv=kfold, return_train_score=True, scoring=acc_scorer)

clf.fit(X, y)



scores = clf.cv_results_['mean_test_score']

scores_std = clf.cv_results_['std_test_score']

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(scores)

clf.best_score_
clf.best_estimator_.get_params()
#note: "**" extracts the parameters

est = SVC(**clf.best_estimator_.get_params())

est.fit(X, y)
data = pd.read_csv('../input/test.csv', index_col=0)



data[data['Age'].isnull() & (data['Name'].str.contains('|'.join(titles)) == False)]

data['Title'] = data.apply(lambda row: NameToTitle(row['Name']), axis=1)

data.update(data[data['Age'].isnull()].apply(lambda row: abt.loc[row['Title']], axis=1))

data['AgeGroup'] = data.apply(lambda row: bisect.bisect_left([0,1,15,65], row.Age) - 1, axis=1)

data.drop('Name', axis=1, inplace=True)

data.drop('Title', axis=1, inplace=True)

data.drop('Age', axis=1, inplace=True)

data['Sex'] = data.apply(lambda row: 0 if(row.Sex == 'male') else 1, axis=1)

data['FareGroup'] = data.apply(lambda row: bisect.bisect_left([0,7,10,11,15,49,52,71,80], row.Fare) - 1, axis=1)

data.drop('Fare', axis=1, inplace=True)

data['TicketNum'] = pd.to_numeric(data.Ticket.str.split(' ').str[-1].replace('LINE', -1))

data['TicketGroup'] = data.apply(lambda row: bisect.bisect_left([0,40000], row.TicketNum) - 1, axis=1)

data.drop('TicketNum', axis=1, inplace=True)

data['Embarked'] = data[['Embarked']].fillna('S').replace({'Embarked': mapping})

data.drop('Cabin', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)

X = data[features]



data["Survived"] = est.predict(X)