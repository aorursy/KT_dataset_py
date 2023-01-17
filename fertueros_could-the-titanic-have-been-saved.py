# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import collections as cln



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn import preprocessing

pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
survived_color = '#6699ff'

died_color = '#ff6666'



na_string = 'NA'

na_number = 0



width = 0.35

embarked_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', na_string: 'N/A'}

pclass_map = {1: 'First class', 2: 'Second class', 3: 'Third class'}



# helper method

def ensure_na(d):

    if not na_string in d:

        d[na_string] = 0

    return d
# Read data and fill NA values with -1



titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# indexes of survived and died 

idx_survived = titanic_df['Survived'] == 1

idx_died = np.logical_not(idx_survived)



titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)



titanic_df["Embarked"] = titanic_df["Embarked"].fillna(na_string)

titanic_df["Fare"] = titanic_df["Fare"].fillna(na_number)

titanic_df["Age"] = titanic_df["Age"].fillna(na_number)

titanic_df["Cabin"] = titanic_df["Cabin"].fillna(na_string)



test_df["Embarked"] = test_df["Embarked"].fillna(na_string)

test_df["Fare"] = test_df["Fare"].fillna(na_number)

test_df["Age"] = test_df["Age"].fillna(na_number)

test_df["Cabin"] = test_df["Cabin"].fillna(na_string)



# features so far

titanic_df[:10]
# Pclass



pclass_survived = titanic_df[idx_survived].Pclass

pclass_died = titanic_df[idx_died].Pclass

pclass_survived_counts = ensure_na(titanic_df[idx_survived].Pclass.value_counts())

pclass_died_counts = ensure_na(titanic_df[idx_died].Pclass.value_counts())



# we get no NA values for Pclass feature

# so we remove NA from plots and sort the rest of values by index

pclass_survived_sorted = pclass_survived_counts[0:3].sort_index()

pclass_died_sorted = pclass_died_counts[0:3].sort_index()



N = len(pclass_survived_sorted)

ind = np.arange(N)



plot1 = plt.bar(ind, pclass_survived_sorted, width, color=survived_color, label='Survived')

plot2 = plt.bar(ind + width, pclass_died_sorted, width, color=died_color, label='Died')



plt.xlabel('Passenger Classes', fontsize=18)

plt.ylabel('Number of people', fontsize=16)

plt.legend(loc='upper center')

plt.xticks(ind + width, (pclass_map[l] for l in pclass_survived_sorted.keys()))

plt.show()



# make dummies from Pclass feature



pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)



titanic_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)
# Sex



sex_dummies_titanic  = pd.get_dummies(titanic_df['Sex'])

sex_dummies_test     = pd.get_dummies(test_df['Sex'])



titanic_df = titanic_df.join(sex_dummies_titanic)

test_df    = test_df.join(sex_dummies_test)



titanic_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



# show features so far

titanic_df[:10]
# Age

age_survived = titanic_df[idx_survived].Age

age_died = titanic_df[idx_died].Age



minAge, maxAge = min(titanic_df.Age), max(titanic_df.Age)

bins = np.linspace(minAge, maxAge, 100)



age_survived_counts, _ = np.histogram(age_survived, bins)

age_died_counts, _ = np.histogram(age_died, bins)



plt.bar(bins[:-1], np.log10(age_survived_counts), color=survived_color, label='Survived')

plt.bar(bins[:-1], -np.log10(age_died_counts), color=died_color, label='Died')

plt.yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))

plt.legend(loc='upper right')

plt.xlabel('Age', fontsize=18)

plt.ylabel('Number of people', fontsize=16)

plt.show()



# Normalization

maxAge = max(titanic_df['Age'])

titanic_df['Age'] /= maxAge

test_df['Age'] /= maxAge



# New feature for Age

titanic_df['AgeNA'] = titanic_df['Age']

titanic_df['AgeNA'].loc[titanic_df['Age'] > 0] = 0

titanic_df['AgeNA'].loc[titanic_df['Age'] <= 0] = 1



test_df['AgeNA'] = test_df['Age']

test_df['AgeNA'].loc[test_df['Age'] > 0] = 0

test_df['AgeNA'].loc[test_df['Age'] <= 0] = 1
# Family members = Parch + SibSp



titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]



# Normalization

maxFamily = max(titanic_df['Family'])

titanic_df['Family'] /= maxFamily

test_df['Family'] /= maxFamily



# print features so far

titanic_df[:10]
# SibSp

sibsp_survived = titanic_df[idx_survived].SibSp

sibsp_died = titanic_df[idx_died].SibSp



sibsp_survived_counts = titanic_df[idx_survived].SibSp.value_counts().sort_index()

sibsp_died_counts = titanic_df[idx_died].SibSp.value_counts().sort_index()



df = pd.concat([sibsp_survived_counts, sibsp_died_counts], axis=1).fillna(0)

print(df)



plot1 = plt.bar(df.index.values - width, np.log10(df.ix[:,0]), width, color=survived_color, label='Survived')

plot2 = plt.bar(df.index.values, np.log10(df.ix[:,1]), width, color=died_color, label='Died')



plt.xlabel('Number of siblings', fontsize=18)

plt.ylabel('Number of people', fontsize=16)

plt.yticks(range(0,4), (10**abs(k) for k in range(0,4)))

plt.xticks(df.index.values)

plt.legend(loc='upper right')

plt.show()



# Normalization

maxSibSp = max(titanic_df['SibSp'])

titanic_df['SibSp'] /= maxSibSp

test_df['SibSp'] /= maxSibSp
# Parch



parch_survived = titanic_df[idx_survived].Parch

parch_died = titanic_df[idx_died].Parch



parch_survived_counts = titanic_df[idx_survived].Parch.value_counts().sort_index()

parch_died_counts = titanic_df[idx_died].Parch.value_counts().sort_index()



df = pd.concat([parch_survived_counts, parch_died_counts], axis=1).fillna(0)

print(df)



plot1 = plt.bar(df.index.values - width, np.log10(df.ix[:,0]), width, color=survived_color, label='Survived')

plot2 = plt.bar(df.index.values, np.log10(df.ix[:,1]), width, color=died_color, label='Died')



plt.xlabel('Number of relatives', fontsize=18)

plt.ylabel('Number of people', fontsize=16)

plt.yticks(range(0,4), (10**abs(k) for k in range(0,4)))

plt.xticks(df.index.values)

plt.legend(loc='upper right')

plt.show()



# Normalization

maxParch = max(titanic_df['Parch'])

titanic_df['Parch'] /= maxParch

test_df['Parch'] /= maxParch



# features so far

titanic_df[:10]
# Fare

fare_survived = titanic_df[idx_survived].Fare

fare_died = titanic_df[idx_died].Fare



minFare, maxFare = min(titanic_df.Fare), max(titanic_df.Fare)

bins = np.linspace(minFare, maxFare, 25)



fare_survived_counts, _ = np.histogram(fare_survived, bins)

fare_died_counts, _ = np.histogram(fare_died, bins)



plt.figure()

plt.bar(bins[:-1], np.log10(fare_survived_counts), width=20, color=survived_color, label='Survived')

plt.bar(bins[:-1], -np.log10(fare_died_counts), width=20, color=died_color, label='Died')

plt.ylabel('Number of people')

plt.xlabel('Ticket fare')

plt.yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))

plt.legend()

plt.show()



# Normalization

maxFare = max(titanic_df['Fare'])

titanic_df['Fare'] /= maxFare

test_df['Fare'] /= maxFare



# New feature for Fare

titanic_df['FareNA'] = titanic_df['Fare']

titanic_df['FareNA'].loc[titanic_df['Fare'] > 0] = 0

titanic_df['FareNA'].loc[titanic_df['Fare'] <= 0] = 1



test_df['FareNA'] = test_df['Fare']

test_df['FareNA'].loc[test_df['Fare'] > 0] = 0

test_df['FareNA'].loc[test_df['Fare'] <= 0] = 1
# Cabin



def removeDigits(col):

    return col.replace('[0-9]+', '', regex=True)



titanic_df['Cabin'] = removeDigits(titanic_df['Cabin'])

test_df['Cabin'] = removeDigits(test_df['Cabin'])



cabin_train = pd.get_dummies(titanic_df['Cabin'])

cabin_test = pd.get_dummies(test_df['Cabin'])



cabin_train.columns = ['cabin_' + name for name in cabin_train.columns]

cabin_test.columns = ['cabin_' + name for name in cabin_test.columns]



train_cols = set(cabin_train.columns)

test_cols = set(cabin_test.columns)



for col in (train_cols - test_cols) | (test_cols - train_cols):

    if col in train_cols:

        cabin_train.drop([col], axis=1, inplace=True)

    if col in test_cols:

        cabin_test.drop([col], axis=1, inplace=True)

    print('Column dropped: %s' % col)

    

titanic_df = titanic_df.join(cabin_train)

test_df    = test_df.join(cabin_test)



titanic_df.drop(['Cabin'], axis=1, inplace=True)

test_df.drop(['Cabin'], axis=1, inplace=True)



# features so far

titanic_df[:10]
# Embarked



survived_embarked_counts = ensure_na(titanic_df[idx_survived].Embarked.value_counts())

died_embarked_counts = ensure_na(titanic_df[idx_died].Embarked.value_counts())

print(survived_embarked_counts)

print(died_embarked_counts)

assert(len(survived_embarked_counts) == len(died_embarked_counts))



N = len(survived_embarked_counts)

ind = np.arange(N) 

plot1 = plt.bar(ind, survived_embarked_counts, width, color=survived_color)

plot2 = plt.bar(ind + width, died_embarked_counts, width, color=died_color)



plt.ylabel('Number of people')

plt.xlabel('Port of Embarkation')

plt.xticks(ind + width, (embarked_map[k] for k in survived_embarked_counts.keys()))

plt.legend((plot1[0], plot2[0]), ('survived', 'died'))

plt.show()



# Embarked train/test get dummies

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])



embark_dummies_titanic.columns = ['port_' + name for name in embark_dummies_titanic.columns]

embark_dummies_test.columns = ['port_' + name for name in embark_dummies_test.columns]



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis=1, inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)



# drop port_NA because it exists in train only

titanic_df.drop(['port_NA'], axis=1, inplace=True)
# train before training

titanic_df[:10]
# test before training

test_df[:10]
assert(all(titanic_df.columns[1:] == test_df.columns[1:]))
X_train = np.array(titanic_df.drop("Survived",axis=1))

Y_train = np.array(titanic_df["Survived"])

X_test  = np.array(test_df.drop("PassengerId",axis=1))



n_train, dim_train = X_train.shape

n_test, dim_test   = X_test.shape



# test

distance = False



# Find the training examples closest to the test examples

if distance:

    import scipy.spatial.distance

    D = scipy.spatial.distance.pdist(np.vstack((np.array(X_train), np.array(X_test))))

    D = scipy.spatial.distance.squareform(D)

    ix = np.argsort(np.amin(D[n_train:, :n_train], axis=0))

    ix = ix[::-1]



    num_trn = 500

    #ix = np.random.permutation(n_train)

    X_trn, Y_trn = X_train[ix[:num_trn]], Y_train[ix[:num_trn]]

    X_val, Y_val = X_train[ix[num_trn:]], Y_train[ix[num_trn:]]



    print("Ratio survived:\n  train: %g\n  trn: %g\n  val: %g" % (

         np.mean(Y_train == 1), np.mean(Y_trn == 1), np.mean(Y_val == 1)))

    print(X_trn.shape)

    print(X_val.shape)



print('Number of training examples: %s, Number of features: %s' % (n_train, dim_train))

print('Number of testing examples:  %s, Number of features: %s' % (n_test, dim_test))

assert(dim_train == dim_test)

print(Y_train.shape)
if distance:

    d = np.sort(np.amin(D[n_train:, :n_train], axis=0))

    plt.plot(np.log10(d))

    d[:50]
# Cross-validation for Random Forest Classifier

userf = False



if userf:

    np.random.seed(5)



    n_folds = 10

    cv = StratifiedKFold(n_folds)

    N_es = [50, 100, 200]

    criteria = ['gini', 'entropy']



    random_forest = RandomForestClassifier()

    gscv = GridSearchCV(estimator=random_forest, param_grid=dict(n_estimators=N_es, criterion=criteria), 

                        n_jobs=1, cv=list(cv.split(X_train, Y_train)), verbose=2)

    gscv.fit(X_train, Y_train)
if userf:

    print('Best CV accuracy: %g\nBest n_estimators: %g\nBest criterion: %s' % (

        gscv.best_score_, gscv.best_estimator_.n_estimators, gscv.best_estimator_.criterion))
# Prediction for Random Forest Classifier

if userf:

    Y_pred = gscv.predict(X_test)



    submission = pd.DataFrame({

            "PassengerId": test_df["PassengerId"],

            "Survived": Y_pred

        })



    file_name = 'Test_predictions_rf_%g_%s.csv' % (gscv.best_estimator_.n_estimators, gscv.best_estimator_.criterion)

    submission.to_csv(file_name, index=False)
# Cross-validation for SVM

useSVM = True

useCV = True



if useSVM:

    n_folds = 10

    cv = StratifiedKFold(n_folds)

    Cs = np.power(2, np.arange(8.0, 14.0))

    kernels = ['rbf']

    gammas = np.power(2, np.arange(-7.0, -2.0))



    svc = SVC()

    if useCV:

        gscv = GridSearchCV(estimator=svc, param_grid=dict(C=Cs, kernel=kernels, gamma=gammas),

                            n_jobs=1, cv=list(cv.split(X_train, Y_train)), verbose=2)

        gscv.fit(X_train, Y_train)

        best_params = gscv.best_params_

    else:

        best_params = {'gamma': 0.015625, 'C': 8192.0, 'kernel': 'rbf'}
if useSVM and useCV:

    print('Best CV accuracy: %g\nBest params: %s\n' % (gscv.best_score_, gscv.best_params_))
from mpl_toolkits.mplot3d import Axes3D



if useSVM and useCV:

    import warnings

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore",category=DeprecationWarning)

        C = [i.parameters['C'] for i in gscv.grid_scores_]

        gamma = [i.parameters['gamma'] for i in gscv.grid_scores_]

        accuracy = [i.mean_validation_score for i in gscv.grid_scores_] 



    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(np.log2(C), np.log2(gamma), accuracy)

    ax.set_xlabel('C')

    ax.set_ylabel('Gamma')

    ax.set_zlabel('Accuracy')



    ax.view_init(30, 120)

    plt.show()
if useSVM:

    # Best CV accuracy: 0.814815, 

    # Best params: {'gamma': 0.015625, 'C': 8192.0, 'kernel': 'rbf'}

    print(best_params)

    svc.set_params(**best_params)

    svc.verbose=True

    

    print(X_train.shape)

    svc.fit(X_train, Y_train)
if useSVM:

    Y_pred = svc.predict(X_test)



    submission = pd.DataFrame({

            "PassengerId": test_df["PassengerId"],

            "Survived": Y_pred

        })



    file_name = 'Test_predictions_svm_rbf_%g_%g.csv' % (svc.C, svc.gamma)

    submission.to_csv(file_name, index=False)
# Nearest Neighbours Classifieer



useNN = False



if useNN:

    print(X_train.shape)

    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=1)

    neigh.fit(X_train, Y_train)

    

    Y_pred = neigh.predict(X_test)

    submission = pd.DataFrame({

            "PassengerId": test_df["PassengerId"],

            "Survived": Y_pred

        })



    file_name = 'Test_predictions_knn_%g.csv' % (neigh.n_neighbors)

    submission.to_csv(file_name, index=False)
print(check_output(["ls", "-alh", "."]).decode("utf8"))
# results

!cat *.csv