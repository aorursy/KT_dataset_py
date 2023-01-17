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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data["Cabin"] = train_data['Cabin'].fillna("Unknown")

train_data.head()
test_data["Cabin"] = test_data['Cabin'].fillna("Unknown")

test_data.head()
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    print(big_string)

    return np.nan



title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']



train_data['Title']=train_data['Name'].map(lambda x: substrings_in_string(x, title_list))

 

#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

train_data['Title']=train_data.apply(replace_titles, axis=1)

train_data.head()
test_data['Title']=test_data['Name'].map(lambda x: substrings_in_string(x, title_list))

test_data['Title']=test_data.apply(replace_titles, axis=1)

test_data.head()
train_data.drop('Name' , axis = 1 , inplace =True)

train_data.head()
test_data.drop('Name' , axis = 1 , inplace =True)

test_data.head()
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_data['Deck']=train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

train_data.head()
test_data['Deck']=test_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

test_data.head()
#Replacing T deck with closest deck G because there is only one instance of T

train_data["Deck"].replace('T' , 'G' , inplace = True)

train_data.drop('Cabin' , axis = 1 , inplace =True)

train_data.head()
test_data.drop('Cabin' , axis = 1 , inplace =True)

test_data.head()
import seaborn as sns

label = train_data["Survived"]

sns.countplot(label)
women = train_data.loc[train_data.Sex == 'female']['Survived']

rate_women = sum(women)/len(women)

print("% rate of women survived",rate_women)
men = train_data.loc[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print("% rate of men survived",rate_men)
import matplotlib.pyplot as plt

fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Sex' , data=train_data , ax = ax[0] , order=['male' , 'female'])

b = sns.countplot(x = 'Sex' , data= train_data[label == 1] , ax = ax[1] , order=['male' , 'female'])

c = sns.countplot(x = 'Sex' , data= train_data[ ((train_data['Age'] < 21) & (label == 1)) ] , order=['male' , 'female'])

ax[0].set_title('All passenger')

ax[1].set_title('Survived passenger')

ax[2].set_title('Survived passenger under age 21')
fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Pclass' , data=train_data , ax = ax[0] , order=[1 ,2,3])

b = sns.countplot(x = 'Pclass' , data=train_data[label == 1] , ax = ax[1] , order=[1 ,2,3])

c = sns.countplot(x = 'Pclass' , data=train_data[ ((train_data['Age'] < 21) & (label == 1)) ] , order=[1,2,3])

ax[0].set_title('All passanger')

ax[1].set_title('Survived passanger')

ax[2].set_title('Survived passanger under age 21')
fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Embarked' , data=train_data , ax = ax[0] , order=['S' ,'Q','C'])

b = sns.countplot(x = 'Embarked' , data= train_data[label == 1] , ax = ax[1] , order=['S' ,'Q','C'])

c = sns.countplot(x = 'Embarked' , data= train_data[ ((train_data['Age'] < 21) & (label == 1)) ] , order=['S' ,'Q','C'])

ax[0].set_title('All passanger')

ax[1].set_title('Survived passanger')

ax[2].set_title('Survived passanger under age 21')
train_data.isna().sum()
test_data.isna().sum()
train_data.loc[train_data.Embarked.isna() , 'Embarked'] = 'S'

train_data.head()
groupMatter = train_data.groupby(['Pclass' , 'Sex' , 'Embarked'])

groupMatter.head()
age_to_fill = groupMatter[['Age']].median()

age_to_fill
for cl in range(1,4):

    for sex in ['male' , 'female']:

        for E in ['C' , 'Q' , 'S']:

            filll = pd.to_numeric(age_to_fill.xs(cl).xs(sex).xs(E).Age)

            train_data.loc[(train_data.Age.isna() & (train_data.Pclass == cl) & (train_data.Sex == sex) 

                    &(train_data.Embarked == E)) , 'Age'] =filll

            test_data.loc[(test_data.Age.isna() & (test_data.Pclass == cl) & (test_data.Sex == sex) 

                    &(test_data.Embarked == E)) , 'Age'] =filll
train_data.isna().sum()
test_data.isna().sum()
train_data.Ticket = pd.to_numeric(train_data.Ticket.str.split().str[-1] , errors='coerce')

train_data
test_data.Ticket = pd.to_numeric(test_data.Ticket.str.split().str[-1] , errors='coerce')

test_data
Ticket_median = train_data.Ticket.median()

train_data.Ticket.fillna(Ticket_median , inplace =True)

train_data.isna().sum()
test_data.Fare.fillna(train_data.Fare.median() , inplace =True)

test_data.isna().sum()
test_data.drop(['Ticket' ] ,axis = 1, inplace = True)

train_data.drop(['Survived','Ticket' ], axis =1, inplace =True )
cat_col = ['Pclass' , 'Sex' , 'Embarked' , 'Title' , 'Deck']

train_data.Pclass.replace({

    1 :'A' , 2:'B' , 3:'C'

} , inplace =True)

test_data.Pclass.replace({

    1 :'A' , 2:'B' , 3:'C'

} , inplace =True)

train_d1 = pd.get_dummies(train_data , columns=cat_col)

test_d1 = pd.get_dummies(test_data , columns=cat_col)

print(train_d1.shape , test_d1.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



train_d1= scaler.fit_transform(train_d1)

train_d1
test_d1 = scaler.transform(test_d1)

test_d1
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

#model = RandomForestClassifier(bootstrap= True , min_samples_leaf= 3, n_estimators = 500 ,

#                               min_samples_split = 10, max_features = "sqrt", max_depth= 6)

#cross_val_score(model , train_d1 , label , cv=5)
from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()

#cross_val_score(model , train_d1 , label , cv=5)
from sklearn.svm import SVC

#model = SVC(C=4)

#cross_val_score(model , train_d1 , label , cv=5)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#model = LinearDiscriminantAnalysis()

#cross_val_score(model, train_d1, label, cv=5)
# Compare Algorithms

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from pprint import pprint



seed = 7



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier()))

#models.append(('CART', DecisionTreeClassifier()))

#models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RForest',RandomForestClassifier()))



results = []

names = []





def gather_results(models,names,results):

    # evaluate each model in turn

    scoring = 'accuracy'

    max = 0

    maxModel=models[0][1]

    for name, model in models:

        kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)

        cv_results = model_selection.cross_val_score(model, train_d1, label, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        if(cv_results.mean() > max):

            max = cv_results.mean()

            maxModel = model

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)



def add_voting(models,names,results):

    eclf = VotingClassifier(estimators=models,voting='hard')

    kfold = model_selection.KFold(n_splits=5, random_state=seed,shuffle=True)



    scores = cross_val_score(eclf, train_d1, label,scoring='accuracy', cv=kfold)

    print("%s: %f (+/- %f)" % ("Voting",scores.mean(), scores.std()))



    names.append('Voting')

    results.append(scores)

    return eclf



def plot(names, results):

    # boxplot algorithm comparison

    fig = plt.figure()

    fig.suptitle('Algorithm Comparison')

    ax = fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()

    

gather_results(models,names,results)

#vote_model = add_voting(models,names,results)

plot(names,results)
#print(models)

# Create first pipeline for base without reducing features.



#from sklearn.pipeline import Pipeline

#pipe = Pipeline([('classifier' , )])

# pipe = Pipeline([('classifier', RandomForestClassifier())])



# Create param grid.

param_grid = []

for name, model in models:

    print('Parameters currently in use for ' + name)

    pprint(model.get_params())

    print('\n')

    

best_models = []

best_names=[]

best_results=[]



#best_models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier()))

#models.append(('CART', DecisionTreeClassifier()))

#models.append(('NB', GaussianNB()))

#best_models.append(('SVM', SVC()))
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV

penalty = ['l1', 'l2']

C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

solver = ['liblinear', 'saga']

hyperparameters = dict(penalty=penalty,

                  C=C,

                  class_weight=class_weight,

                  solver=solver)

gridsearch = GridSearchCV(models[0][1], hyperparameters, cv=5, verbose=1) # Fit grid search

random_search = RandomizedSearchCV(estimator = models[0][1], param_distributions = hyperparameters, n_iter = 100, cv = 5, verbose=1, random_state=42, n_jobs = -1)

best_model_lr = random_search.fit(train_d1, label)

print('Best params:', best_model_lr.best_estimator_.get_params())

label_lr = 'LR'

score_lr = best_model_lr.score(train_d1, label)

print("The mean accuracy of the model is:",score_lr)

#best_models.append((label_lr, (best_model_lr)))

#best_names.append(label_lr)

#best_results.append(score_lr)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

hyperparameters = dict(n_estimators=n_estimators,

                  max_features=max_features,

                      max_depth=max_depth,

                      min_samples_split=min_samples_split,

                      min_samples_leaf=min_samples_leaf,

                       bootstrap=bootstrap)

gridsearch = GridSearchCV(models[3][1], hyperparameters, cv=5, verbose=1) # Fit grid search

random_search = RandomizedSearchCV(estimator = models[3][1], param_distributions = hyperparameters, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)

best_model_forest = random_search.fit(train_d1, label)

print('Best params:', best_model_forest.best_estimator_.get_params())

label_forest = 'RForest'

score_forest = best_model_forest.score(train_d1, label)

print("The mean accuracy of the model is:",score_forest)

best_models.append((label_forest,best_model_forest))

best_names.append(label_forest)

best_results.append(score_forest)
#vote_model = add_voting(best_models,best_names,best_results)

print("Best names: ",best_names)

print("Best results: ",best_results)

plot(best_names,best_results)
maxModel = best_model_forest

maxModel.fit(train_d1, label)
score_max = maxModel.score(train_d1, label)

print("The mean accuracy of the model is:",score_max)

predictions = maxModel.predict(test_d1)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.head()

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")