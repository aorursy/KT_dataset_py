import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import math

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer, accuracy_score
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train_data
train_data[['Last','First']] = train_data.Name.str.split(pat=',', expand=True)

train_data[['Title','First']] = train_data.First.str.split(pat='.', n=1, expand=True)

train_data[['Title']] = train_data.Title.str.replace(' ', '')

print(train_data.Title.unique())

train_data = train_data.drop(['First', 'Last', 'Name'], axis=1)



test_data[['Last','First']] = test_data.Name.str.split(pat=',', expand=True)

test_data[['Title','First']] = test_data.First.str.split(pat='.', n=1, expand=True)

test_data[['Title']] = test_data.Title.str.replace(' ', '')

test_data = test_data.drop(['First', 'Last', 'Name'], axis=1)
first_c = train_data.loc[train_data['Pclass'] == 1]

second_c = train_data.loc[train_data['Pclass'] == 2]

third_c = train_data.loc[train_data['Pclass'] == 3]



first_c_dat = first_c.groupby('Title').PassengerId.nunique().to_frame('count').reset_index()

second_c_dat = second_c.groupby('Title').PassengerId.nunique().to_frame('count').reset_index()

third_c_dat = third_c.groupby('Title').PassengerId.nunique().to_frame('count').reset_index()

third_c_dat
fig, ax = plt.subplots(figsize=(15, 5), ncols=3)



first_c_titles = first_c_dat["Title"].tolist()

first_c_count = first_c_dat["count"].tolist()

ax[0].bar(first_c_titles, first_c_count, align='center', color="bisque", edgecolor="black")

#ax[0].set(xticks=range(10), xlim=[-1, 10])

ax[0].set_title('Titles in first class')

ax[0].set_ylabel("count")

ax[0].set_xlabel("Title")



second_c_titles = second_c_dat["Title"].tolist()

second_c_count = second_c_dat["count"].tolist()

ax[1].bar(second_c_titles, second_c_count, align='center', color="burlywood", edgecolor="black")

#ax[1].set(xticks=range(10), xlim=[-1, 10])

ax[1].set_title('Titles in second class')

ax[1].set_xlabel("Title")



third_c_titles = third_c_dat["Title"].tolist()

third_c_count = third_c_dat["count"].tolist()

ax[2].bar(third_c_titles, third_c_count, width=0.8, align='center', color="darkgoldenrod", edgecolor="black")

#ax[2].set(xticks=range(10), xlim=[-1, 10])

ax[2].set_title('Titles in third class')

ax[2].set_xlabel("Title")

plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=-90 )

plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=-40 )

plt.show()
print(train_data.SibSp.unique())

print(train_data.Parch.unique())
sibsp_dat = train_data.groupby(['SibSp', 'Survived']).PassengerId.nunique().to_frame('count').reset_index()

sibsp_surv = sibsp_dat.loc[sibsp_dat['Survived'] == 1]

sibsp_dec = sibsp_dat.loc[sibsp_dat['Survived'] == 0]



ind = np.arange(9)

plt.bar(sibsp_surv['SibSp'].tolist(), sibsp_surv['count'], width=0.5, label='Survived', color="palegreen", edgecolor="black")

plt.bar([x + 0.5 for x in sibsp_dec['SibSp'].tolist()], sibsp_dec['count'], width=0.5, label='Deceased', color="lightcoral", edgecolor="black")

plt.legend(loc='best')

plt.xlabel("number of siblings and spouses on board")

plt.ylabel('count')

plt.title('Number of siblings/spouses on board vs survival')

plt.xticks(ind + 0.5 / 2, ('0', '1', '2', '3', '4', '5', '6', '7', '8'))

plt.show()
parch_dat = train_data.groupby(['Parch', 'Survived']).PassengerId.nunique().to_frame('count').reset_index()

parch_surv = parch_dat.loc[parch_dat['Survived'] == 1]

parch_dec = parch_dat.loc[parch_dat['Survived'] == 0]



ind = np.arange(7)

plt.bar(parch_surv['Parch'].tolist(), parch_surv['count'], width=0.5, label='Survived', color="palegreen", edgecolor="black")

plt.bar([x + 0.5 for x in parch_dec['Parch'].tolist()], parch_dec['count'], width=0.5, label='Deceased', color="lightcoral", edgecolor="black")

plt.legend(loc='best')

plt.xlabel("number of parents and children on board")

plt.ylabel('count')

plt.title('Number of parents/children on board vs survival')

plt.xticks(ind + 0.5 / 2, ('0', '1', '2', '3', '4', '5', '6'))

plt.semilogy()

plt.show()
# print(train_data.Ticket.unique())

# print(train_data.Fare.unique())

# print(train_data.Cabin.unique())
train_data['Floor'] = train_data['Cabin'].astype(str).str[0]

train_data = train_data.drop(['Cabin'], axis=1)



test_data['Floor'] = test_data['Cabin'].astype(str).str[0]

test_data = test_data.drop(['Cabin'], axis=1)
def f(x):

    if x['Pclass'] == 1 and x['Floor'] == 'n': return 'C'

    elif x['Pclass'] == 2 and x['Floor'] == 'n': return 'E'

    elif x['Pclass'] == 3 and x['Floor'] == 'n': return 'F'

    else: return x['Floor']



cabin_dat = train_data.groupby(['Pclass', 'Floor']).PassengerId.count().to_frame('Count').reset_index()

print(cabin_dat)

# 1: C

# 2: E

# 2: F

train_data['Floor'] = train_data.apply(f, axis=1)

test_data['Floor'] = test_data.apply(f, axis=1)



cabin_dat = train_data.groupby(['Pclass', 'Floor']).PassengerId.count().to_frame('Count').reset_index()

print(cabin_dat)
fare_dat = train_data.groupby(by=['Pclass', pd.cut(train_data['Fare'], bins=np.linspace(-10, 540, 54))]).PassengerId.count().to_frame('count').reset_index()

fir_fare_dat = fare_dat.loc[fare_dat['Pclass'] == 1]

sec_fare_dat = fare_dat.loc[fare_dat['Pclass'] == 2]

thi_fare_dat = fare_dat.loc[fare_dat['Pclass'] == 3]

# 1: 0-512.3292

# 2: 10-73.50

# 3: 0-69.55



fig, ax = plt.subplots(figsize=(15, 5), ncols=3)



# first_c_fare = [lambda x: x.left in fir_fare_dat["Fare"]]

first_c_fare = fir_fare_dat["Fare"].tolist()

first_c_fare = [x.right for x in first_c_fare] 

first_c_count = fir_fare_dat["count"].tolist()

ax[0].bar(first_c_fare, first_c_count, width=10, align='edge', color="bisque", edgecolor="black")

ax[0].set(xticks=range(0, 540, 50), xlim=[0, 540])

ax[0].set_title('Fare in first class')

ax[0].set_ylabel("count")

ax[0].set_xlabel("Fare")



second_c_fare = sec_fare_dat["Fare"].tolist()

second_c_fare = [x.right for x in second_c_fare] 

second_c_count = sec_fare_dat["count"].tolist()

ax[1].bar(second_c_fare, second_c_count, width=10, align='edge', color="burlywood", edgecolor="black")

ax[1].set_title('Fare in second class')

ax[1].set_xlabel("Fare")



third_c_fare = thi_fare_dat["Fare"].tolist()

third_c_fare = [x.right for x in third_c_fare] 

third_c_count = thi_fare_dat["count"].tolist()

ax[2].bar(third_c_fare, third_c_count, width=10, align='edge', color="darkgoldenrod", edgecolor="black")

ax[2].set_title('Fare in third class')

ax[2].set_xlabel("Fare")

plt.show()
train_data["Embarked"] = train_data["Embarked"].fillna("S")

test_data["Embarked"] = test_data["Embarked"].fillna("S")



emb_dat = train_data.groupby(['Embarked', 'Survived']).PassengerId.nunique().to_frame('count').reset_index()

emb_surv = emb_dat.loc[emb_dat['Survived'] == 1]

emb_dec = emb_dat.loc[emb_dat['Survived'] == 0]



ind = np.arange(3)

plt.bar(ind, emb_surv['count'], width=0.5, label='Survived', color="palegreen", edgecolor="black")

plt.bar(ind + 0.5, emb_dec['count'], width=0.5, label='Deceased', color="lightcoral", edgecolor="black")

plt.legend(loc='best')

plt.xlabel("Entry Port")

plt.ylabel('count')

plt.title('Entry port and chance of survival')

plt.xticks(ind + 0.5 / 2, ('Cherbourg', 'Queenstown', 'Southampton'))

plt.show()
# Filling in missing age data

def age(x, mu, sigma):

    return np.random.randint(low=mu-sigma, high=mu+sigma)



train_mean, train_std = train_data['Age'].std(), train_data['Age'].std()

train_data['Age'] = train_data.apply(age, args=(train_mean, train_std), axis=1)

test_mean, test_std = test_data['Age'].std(), test_data['Age'].std()

test_data['Age'] = test_data.apply(age, args=(test_mean, test_std), axis=1)

# convert from float to int

train_data['Age'] = train_data['Age'].astype(int)

test_data['Age'] = test_data['Age'].astype(int)



# Missing Fare data

#13.91 was the average fare for the one passenger in the test df riding third class and embarked at Southhampton

test_data["Fare"][np.isnan(test_data["Fare"])] = 13.91



# Drop extra columns

# train_data = train_data.drop(['Ticket', 'PassengerId'], axis=1)

# test_data = test_data.drop(['Ticket', 'PassengerId'], axis=1)



# Floor has an order and may be changed to numerical values

def floor(x):

    new_floors = {"T": 7, "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}

    return new_floors[x.Floor]



train_data['Floor'] = train_data.apply(floor, axis=1)

test_data['Floor'] = test_data.apply(floor, axis=1)



# One Hot Encoding for Sex, Embarked, and Title because there is no order to these categories.

sex_dum_train = pd.get_dummies(train_data['Sex'])

sex_dum_test = pd.get_dummies(test_data['Sex'])

train_data = train_data.join(sex_dum_train)

test_data = test_data.join(sex_dum_test)



embark_dum_train = pd.get_dummies(train_data['Embarked'])

embark_dum_test = pd.get_dummies(test_data['Embarked'])

train_data = train_data.join(embark_dum_train)

test_data = test_data.join(embark_dum_test)



title_dum_train = pd.get_dummies(train_data['Title'])

title_dum_test = pd.get_dummies(test_data['Title'])

train_data = train_data.join(title_dum_train)

test_data = test_data.join(title_dum_test)



# Drop extra columns

train_data = train_data.drop(['Ticket', 'PassengerId', 'Sex', 'Embarked', 'Title'], axis=1)

test_data = test_data.drop(['Ticket', 'PassengerId', 'Sex', 'Embarked', 'Title'], axis=1)





# columns in training data must match the columns in testing data

# test needs a Capt, Don, Jonkheer, Lady, Major, Mlle, Mme, Sir, theCountess  column; train needs a Dona column

test_data['Capt'] = 0

test_data['Don'] = 0

test_data['Jonkheer'] = 0

test_data['Lady'] = 0

test_data['Major'] = 0

test_data['Mlle'] = 0

test_data['Mme'] = 0

test_data['Sir'] = 0

test_data['theCountess'] = 0



train_data['Dona'] = 0
fold0, fold1 = np.array_split(train_data, 2) 

fold0_tar, fold1_tar = fold0['Survived'], fold1['Survived']

fold0, fold1 = fold0.drop(['Survived'], axis=1), fold1.drop(['Survived'], axis=1)

fold_list = [{"data": fold0, "target": fold0_tar}, {"data": fold1, "target": fold1_tar}] 
def get_random_grid():

    # 'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True

    

    # Number of trees in random forest

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split

    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree

    max_depth = [int(x) for x in np.linspace(18, 24, num = 7)]

    max_depth.append(None)

    # Minimum number of samples required to split a node

    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node

    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree

    bootstrap = [True, False]



    # Create the random grid

    random_grid = {'n_estimators': n_estimators,

                   'max_features': max_features,

                   'max_depth': max_depth,

                   'min_samples_split': min_samples_split,

                   'min_samples_leaf': min_samples_leaf,

                   'bootstrap': bootstrap}



    print(random_grid)

    return random_grid



def random_search(train_data, target_data):

    weights_list = []

    clf = RandomForestClassifier(bootstrap=True, n_estimators=200, max_depth=7, 

                                 random_state=0)

    random_grid = get_random_grid()

    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,

                          n_iter = 200, #scoring='neg_mean_absolute_error', 

                          cv = 3, verbose=2, random_state=42, n_jobs=-1,

                          return_train_score=True)



    clf_random.fit(train_data, target_data)  

    return clf_random.cv_results_

#     predictions = clf_random.predict()

#     accuracy = accuracy_score(target_data, predictions)

#     return weights_list, accuracy

#         # print(clf.predict([[0, 0, 0, 0]]))



def train(train_data, target_data, parameters):

#     clf = RandomForestClassifier(bootstrap=True, max_depth=22, random_state=0, n_estimators=1200, 

#                                  min_samples_split=2, min_samples_leaf=2, max_features='auto')

    clf = RandomForestClassifier(**parameters)

    kf = KFold(n_splits=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(train_data):

        fold += 1

        X_train, X_test = train_data.values[train_index], train_data.values[test_index]

        y_train, y_test = target_data.values[train_index], target_data.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 

    return clf
X = train_data.drop(['Survived'], axis=1)

y = train_data['Survived']

random_results = random_search(X, y)
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        top_result = True

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")

            if top_result:

                best_params = results['params'][candidate]

    return best_params



best_parameters = report(random_results)
results = train(X, y, best_parameters)
weights_list = results.feature_importances_
train_data.columns.shape[0]

weights_list.shape
fig, ax = plt.subplots(figsize=(15, 5), ncols=1)



columns = np.arange(weights_list.shape[0])

ax.bar(columns, weights_list, align='center', color="lightcyan", edgecolor="black")

ax.set(xticks=range(29))

ax.set_title('Weights for first fold')

ax.set_ylabel("count")

ax.set_xlabel("Title")

ax.set_xticklabels(train_data.columns.tolist())



# columns = np.arange(weights_list[1].shape[0])

# ax[1].bar(columns, weights_list[1], align='center', color="lightcyan", edgecolor="black")

# #ax[1].set(xticks=range(10), xlim=[-1, 10])

# ax[1].set_title('Weights for second fold')

# ax[1].set_xlabel("Title")

# ax[1].set(xticks=range(29))

# ax[1].set_xticklabels(train_data.columns.tolist())



# for plot in ax:

#     plt.sca(plot)

plt.xticks(rotation=90)

plt.show()
test_predictions = results.predict(test_data)  
test_predictions

output = gender_submission.assign(Survived=test_predictions)

output
output.to_csv('submission.csv', index=False)