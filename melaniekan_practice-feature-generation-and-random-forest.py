import pandas as pd

import numpy as np



import plotly.offline as pyoff

from plotly.graph_objs import *



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor



from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV
# dataset with survival

fname = '../input/train.csv'

Xgiven = pd.read_csv(fname)

Xgiven = Xgiven.drop(labels = 'PassengerId', axis = 1)

NX, Nfeature = Xgiven.shape
# dataset for prediction

fname = '../input/test.csv'

Xpredict = pd.read_csv(fname)

result = pd.DataFrame(Xpredict.PassengerId)

Xpredict = Xpredict.drop(labels = 'PassengerId', axis = 1)
# combine two datasets for feature engineering

X0 = pd.concat([Xgiven.drop(labels = 'Survived', axis = 1), Xpredict],

                axis = 0, ignore_index = True)

Nentry, Nfeature = X0.shape
# code sex as numbers

y = X0.Sex

X0.Sex = y.map({'male':0, 'female':1})
X0.Sex.unique()
# code embark as numbers

y = X0.Embarked

X0.Embarked = y.map({ 'S':1, 'C':2, 'Q':3 })
# since the most passengers embark at 'S', change NaN to 1

y = X0.Embarked

X0.loc[y.isnull(), 'Embarked'] = 1.0
X0.Embarked.unique()
# fill up NaN by the mean value of the corresponding class

idc = X0.Fare.isnull()

iclass = X0.Pclass.loc[idc].values[0]

fare = X0.loc[X0.Pclass == iclass, 'Fare']

fare = fare.mean()

X0.loc[idc,'Fare'] = fare
y = X0.Name.str.extract('(\w+)\.', expand = False)
# calculate the survival rate for each title for categorization

titles = y.unique()

survive = np.zeros(len(titles))

cnt = 0

for ititle in titles:

    idc = y == ititle

    isurvive = Xgiven.Survived[idc].mean()

    survive[cnt] = isurvive

    cnt += 1
# plot: survival v.s. title

data = [Bar(x = titles, y = survive, opacity = 0.6,

            marker = dict(color = 'rgb(158,202,225)',

                          line = dict(color = 'rgb(8,48,107)', width = 1.5)))]

layout = Layout(title = 'Survival rate v.s. passenger title',

                autosize = False, width = 700, height = 400)

fig = Figure(data = data, layout = layout)

pyoff.init_notebook_mode(connected = True)

pyoff.iplot(fig)
Y = pd.DataFrame(y)

Y.columns = ['Title']

rules = {'Don': 'Mr', 'Dona': 'Mrs', 'Mme': 'Mrs', 'Ms': 'Mrs', 'Mlle':'Miss',

         'Master':'Level1', 'Dr':'Level1', 'Major':'Level1', 'Col':'Level1', 'Jonkheer':'Level1',

         'Lady':'Level2', 'Sir':'Level2', 'Countess':'Level2',

         'Rev': 'Charge', 'Capt': 'Charge'}

Y.Title = Y.replace({'Title': rules})

X0 = pd.concat([X0, Y], axis = 1)

X0.Title = X0.Title.map({'Charge':1, 'Mr':2, 'Level1':3, 'Miss':4, 'Mrs':5, 'Level2':6})
X0.Title.unique()
cabins = ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

NAidc = X0.Cabin.isnull()

y = np.zeros(Nentry)

for i in range(Nentry):

    if NAidc[i] == False: # code the cabins as numbers

        icabin = X0.Cabin.iloc[i][0]

        y[i] = cabins.index(icabin)

    else: # also code those with NaN values

        iclass = X0.Pclass.iloc[i]

        y[i] = 7 + iclass

X0.Cabin = y
X0.Cabin.unique()
y = X0.SibSp + X0.Parch + 1

Y = pd.DataFrame(y, columns = ['FamilySize'])

X0 = pd.concat([X0, Y], axis = 1)
# plot survival rate v.s. FamilySize for banding

X = pd.concat([Xgiven.Survived, X0.iloc[:NX, :]], axis = 1)

y = X[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

# plot: survival v.s. title

data = [Bar(x = y.FamilySize, y = y.Survived, opacity = 0.6,

            marker = dict(color = 'rgb(158,202,225)',

                          line = dict(color = 'rgb(8,48,107)', width = 1.5)))]

layout = Layout(title = 'Survival rate v.s. family size',

                autosize = False, width = 700, height = 400)

fig = Figure(data = data, layout = layout)

pyoff.init_notebook_mode(connected = True)

pyoff.iplot(fig)
# create a band summarizing FamilySize

y = np.zeros(Nentry)

for i in range(Nentry):

    if X0.FamilySize[i] == 1: continue;

    elif X0.FamilySize[i] > 1 and X0.FamilySize[i] <= 4: y[i] = 1

    elif X0.FamilySize[i] > 4 and X0.FamilySize[i] <= 7: y[i] = 2

    else: y[i] = 3

# combine into data

Y = pd.DataFrame(y, columns = ['FamilyBand'])

X0 = pd.concat([X0, Y], axis = 1)
X0.FamilyBand.unique()
# check the correlation between features

X = X0.drop(labels = ['Name', 'Ticket'], axis = 1)

cors = X.corr()

cors2 = cors.as_matrix()

cors2 = abs(cors2)

data = [Heatmap(z = cors2, x = cors.columns, y = cors.columns, colorscale = 'YlGnBu')]

pyoff.iplot(data)
# predict the ages using the given values



# separate into target and training set

X2 = X0.drop(labels = ['Name', 'Ticket'], axis = 1)

idc = X2.Age.isnull()

X_target = X2.loc[idc, :] # those with NaNs

X_target = X_target.drop(labels = 'Age', axis = 1)

# separate into labels & data

X_given = X2.loc[idc == False, :]

Y_given = X_given.Age # 'labels'

X_given = X_given.drop(labels = 'Age', axis = 1)



# Random forest regression

rfr = RandomForestRegressor(criterion = 'mse')

param_dist = {"n_estimators": sp_randint(5, 25),

              "max_features": sp_randint(2, len(X_given.columns)+1),

              "max_depth": [3, 51],

              "min_samples_leaf": sp_randint(1, 11)}

random_search = RandomizedSearchCV(rfr, param_distributions = param_dist, n_iter = 10)

random_search.fit(X_given, Y_given)



print('score: ', random_search.best_score_)

print('Parameters: ', random_search.best_params_)
# replace by prediction

ages = random_search.best_estimator_.predict(X_target.values)

idc = X0.Age.isnull()

X0.loc[idc, 'Age'] = ages
# bucket Age into bins

y, bins = pd.cut(X0.Age, 5, retbins = True)

y = np.zeros(Nentry)

for i in range(Nentry):

    for j in range(len(bins)-1):

        if X0.Age[i] > bins[j] and X0.Age[i] <= bins[j+1]:

            y[i] = j

            break

Y = pd.DataFrame(y, columns = ['AgeBand'])

X0 = pd.concat([X0, Y], axis = 1)
X0.AgeBand.unique()
# replace by modified data

X = pd.concat([Xgiven.Survived, X0.iloc[:NX, :]], axis = 1)
X.head()
# separate into labels & data

X2 = X.drop(labels = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis = 1)

Y = X2.Survived

X2 = X2.drop(labels = ['Survived'], axis = 1)

# separate into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size = .1)
# Random forest classifier

rfc = RandomForestClassifier(criterion = 'mse')

param_dist = {"n_estimators": sp_randint(5, 25),

              "criterion": ['gini', 'entropy'],

              "max_features": sp_randint(2, len(X2.columns)+1),

              "max_depth": [3, 51],

              "min_samples_leaf": sp_randint(1, 11)}

random_search = RandomizedSearchCV(rfc, param_distributions = param_dist, n_iter = 10)

random_search.fit(X_train, Y_train)



print('score: ', random_search.best_score_)

print('Parameters: ', random_search.best_params_)



# prediction score

print('validation score: ', random_search.best_estimator_.score(X_test, Y_test))
X = X0.iloc[NX:, :]

X2 = X.drop(labels = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis = 1)

y = random_search.best_estimator_.predict(X2)
Y = pd.DataFrame(y, columns = ['Survived'])

result = pd.concat([result, Y], axis = 1)

result.to_csv('titanic_prediction.csv', index = False)