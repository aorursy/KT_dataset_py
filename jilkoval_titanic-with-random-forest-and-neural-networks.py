import pandas as pd
import numpy as np
import random as rnd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# read data
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")
train_raw.head()
print(train_raw.columns)
train_raw.describe(include = "all")
train_raw.info()
test_raw.describe(include = "all")
test_raw.info()
# missing values in train
train_raw.isnull().sum()
# missing values in test
test_raw.isnull().sum()
# deep copy to copy data and indeces
train = train_raw.copy(deep=True)
test = test_raw.copy(deep=True)

# dropping Passenger's IDs and Cabin
train = train.drop(['PassengerId', 'Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train['Age'] = train['Age'].fillna(-1)
test['Age'] = test['Age'].fillna(-1)
train[train['Embarked'].isnull()]
(pd.concat([train,test])).groupby('Embarked').count()['Sex']
train['Embarked'] = train['Embarked'].fillna('S')
test[test['Fare'].isnull()]
sns.pairplot(train, hue="Survived", dropna=True)
corr = train.corr()
sns.heatmap(corr, square=True, annot=True, center=0)
mean_fare_3 = np.nanmean(pd.concat([train[train['Pclass']==3]['Fare'],test[test['Pclass']==3]['Fare']]).values)
print("Mean Fare of passengers with Pclass==3:", mean_fare_3)
test['Fare'] = test['Fare'].fillna(mean_fare_3)
train.isnull().sum()
test.isnull().sum()
train["Ticket"][:20]
pd.concat(g for _, g in train.groupby("Ticket") if len(g) > 1)[:20]
train['Alone'] = 1
train.loc[train.duplicated(subset='Ticket'),'Alone'] = 0
sns.factorplot(x="Alone", y="Survived", data=train)
test['Alone'] = 1
test.loc[test.duplicated(subset='Ticket'),'Alone'] = 0
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
# the following is adopted from 
# https://www.kaggle.com/niklasdonges/end-to-end-project-with-python

data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
pd.concat([train,test]).groupby('Title').count()['Sex']
sns.factorplot(x="Title", y="Survived", data=train)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
sns.factorplot(x="SibSp", y="Survived", data=train)
sns.factorplot(x="Parch", y="Survived", data=train)
train['FamSize'] = train["Parch"].values+train["SibSp"]
sns.factorplot(x='FamSize', y="Survived", data=train)
corr = train.corr()
sns.heatmap(corr, square=True, annot=True, center=0)
test['FamSize'] = test["Parch"].values+test["SibSp"]
pd.concat([train,test]).groupby('Sex').count()['Age']
pd.concat([train,test]).groupby('Title').count()['Age']
pd.concat([train,test]).groupby('Alone').count()['Age']
pd.concat([train,test]).groupby('Embarked').count()['Age']
pd.concat([train,test]).groupby('Pclass').count()['Age']
data = [train, test]
dummies = ['Pclass', 'Sex', 'Embarked', 'Alone', 'Title']

# keeps the original variable as well
train = pd.concat([train, pd.get_dummies(train[dummies], columns=dummies)], axis=1)
test = pd.concat([test, pd.get_dummies(test[dummies], columns=dummies)], axis=1)

# replaces the original variable by the dummy columns
#train = pd.get_dummies(train, columns=dummies)
#test = pd.get_dummies(test, columns=dummies)
train.columns
plt.hist([train.loc[train['Survived']==0,'Age'].values, train.loc[train['Survived']==1,'Age'].values], 
         color=['r','b'], 
         alpha=0.5,
         label=['survived=0','survived=1'])
plt.legend()
plt.xlabel('Age')
plt.ylabel('#')
f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize=(15,5))
data_plot = [train.loc[train['Survived']==0,'Age'].values, train.loc[train['Survived']==1,'Age'].values]
hist_arg = {'color': ['r','b'], 
            'alpha': 0.5,
            'label':['survived=0','survived=1']}
for bin_size,ax in zip([5,10,15,20],[ax1,ax2,ax3,ax4]):
    ax.hist(data_plot, 
            bins=range(-bin_size,90,bin_size),
            **hist_arg)
plt.legend()
plt.xlabel('Age')
plt.ylabel('#')
# this is binning used by Li-Yen Hsu in his/her kernel: Titanic - Neural Network
# in https://www.kaggle.com/liyenhsu/titanic-neural-network

plt.hist(data_plot, 
         bins=[ 0, 4, 12, 18, 30, 50, 65, 100],
         **hist_arg)
age_0 = train.loc[train['Survived']==0,'Age'].values
age_1 = train.loc[train['Survived']==1,'Age'].values

for bin_size in [5,10,15,20]:
    bins_0 = np.histogram(age_0,bins=range(-bin_size,90,bin_size), range=(-bin_size,80))
    bins_1 = np.histogram(age_1,bins=range(-bin_size,90,bin_size), range=(-bin_size,80))
    print(bins_1[0]/(bins_0[0]+bins_1[0]))

# and Li-Yen Hsu binning again
bins = [ 0, 4, 12, 18, 30, 50, 65, 100]
bins_0 = np.histogram(age_0,bins=bins)
bins_1 = np.histogram(age_1,bins=bins)
print(bins_1[0]/(bins_0[0]+bins_1[0]))
bins_age = [ 0, 4, 12, 18, 30, 50, 65, 100]
lab_ages = [0,1,2,3,4,5,6]
train.columns
cols_for_age_corr = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'Alone', 'FamSize']
corr = (train[cols_for_age_corr]).corr()
sns.heatmap(corr, square=True, annot=True, center=0)
grid = sns.FacetGrid(train, col='Title', sharey=False)
grid.map(plt.hist, 'Age', bins=range(0,105,5))
plt.show()
data_all = pd.concat([train,test])
print('mean', data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).mean())
print('std', data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).std())
grid = sns.FacetGrid(train, col='Pclass', sharey=False)
grid.map(plt.hist, 'Age', bins=range(0,105,5))
plt.show()
plt.scatter(train['Age'].values, train['Fare'])
# counts of missing age values for different titles
train.loc[train['Age']<0].groupby('Title').count()['Sex']
plt.scatter(train['Age'].values, train['SibSp'])
means = data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).mean()
stds = data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).std()

np.random.seed(seed=666)
data = [train,test]
for data_i in data:
    ages_i = []
    for index, row in data_i[data_i['Age']<0].iterrows():
        mu = means.loc[means.index==row['Title'],'Age'].values
        std = stds.loc[stds.index==row['Title'],'Age'].values
        ages_i.append(np.random.normal(mu,std)[0])
    #print(ages_i)
    data_i.loc[data_i['Age']<0,'Age'] = ages_i
for data_i in data:
    data_i['Age_bin'] = pd.cut(data_i.Age,bins_age,labels=lab_ages)
train = pd.concat([train, pd.get_dummies(train['Age_bin'], columns=['Age_bin'], prefix='Age_bin', prefix_sep='_')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Age_bin'], columns=['Age_bin'], prefix='Age_bin', prefix_sep='_')], axis=1)
n_on_ticket = []
for index, row in train_raw.iterrows():
    n_on_ticket.append(1.*sum(train_raw["Ticket"]==row['Ticket']))
fare_per_person = train_raw['Fare'].values/np.array(n_on_ticket)
train_raw['Fare_pp'] = fare_per_person
print(train_raw.corr())
train['Fare_pp'] = train_raw['Fare_pp']
# and now for test set
n_on_ticket = []
for index, row in test_raw.iterrows():
    n_on_ticket.append(1.*sum(test_raw["Ticket"]==row['Ticket']))
test['Fare_pp'] = test['Fare'].values/np.array(n_on_ticket)
# list of all features
train.columns
features_to_use = ['Pclass_1', 'Pclass_2', 'Pclass_3', 
                   'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S', 
                   'Alone_0', 'Alone_1', 
                   'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 
                   'Age_bin_0', 'Age_bin_1', 'Age_bin_2', 'Age_bin_3', 'Age_bin_4', 'Age_bin_5', 'Age_bin_6', 
                   'Fare_pp', 'SibSp', 'Parch']

# training data
X_all = train[features_to_use]
y_all = train['Survived']

# test data
X_test = test[features_to_use]
# test ID to save result
pr_id = test_raw['PassengerId']
from scipy.stats import randint

# Randomized search on RF hyper parameters
# specify parameters and distributions to sample from
param_dist_rf = {"max_depth": randint(5, 30),
                 "min_samples_split": randint(2, 11),
                 "min_samples_leaf": randint(1, 6),
                 "max_leaf_nodes": randint(10, 50),
                 "criterion": ["gini", "entropy"],
                 "n_estimators": randint(100, 1000)}

# Utility function to report best scores
# see http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
from sklearn.model_selection import RandomizedSearchCV

# wrap-up function for hyper-parameter tuning of general ML classifier
def model_random_search(cl, param_dist, n_iter, X, y):
    random_search = RandomizedSearchCV(cl, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       random_state=666,
                                       cv=4)
                                       #verbose=2)
    random_search = random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)
    return random_search
from sklearn.ensemble import RandomForestClassifier

# less iterations used for shorter running time
#random_search_rf1 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 30, X_all, y_all)
random_search_rf1 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 5, X_all, y_all)
# some helper functions

def save_prediction(answer, file_out):
    np.savetxt(file_out, answer, header='PassengerId,Survived', delimiter=',', fmt='%i', comments='')
    
def apply_cl(cl, xtest, pr_id):
    pr_test_data = cl.predict(xtest)
    
    answer = np.array([pr_id,pr_test_data]).T
    #print(np.shape(answer))
    return answer

def recall(y_hat, y_obs):
    true_pos = y_obs*y_hat
    return np.sum(true_pos)/np.sum(y_obs)

def precission(y_hat, y_obs):
    true_pos = y_obs*y_hat
    return np.sum(true_pos)/np.sum(y_hat)
    
def train_cl_param(cla, X, y, param):
    cl = cla(**param)
    cl.fit(X,y)
    try: score = cl.oob_score_
    except: score = cl.score(X,y)
    r = recall(cl.predict(X), y)
    p = precission(cl.predict(X), y)
    f = 2.*(p*r)/(p+r)
    print('precission =', p, 'recall =', r, 'f-score =', f, 'score', score)
    return cl

def write_result(cl, file_out, X_test, pr_id):
    an = apply_cl(cl, X_test, pr_id)
    save_prediction(an, file_out)
par = {'max_leaf_nodes': 25, 
       'min_samples_split': 9, 
       'min_samples_leaf': 3, 
       'criterion': 'entropy', 
       'max_depth': 28, 
       'n_estimators': 557,
       'n_jobs':1,
       'oob_score':True}
rf = train_cl_param(RandomForestClassifier, X_all, y_all, par)
#write_result(rf, 'random_forest_3.csv', X_test, pr_id)
def get_importances(cl, features):
    importances = cl.feature_importances_
    std = np.std([cl.feature_importances_ for tree in cl.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    for i in range(len(features)):
        print(features_to_use[indices[i]], importances[indices[i]], std[indices[i]])
get_importances(rf, features_to_use)
# based on some discussions on Kaggle 
# (e.g. Oscar Takeshita https://www.kaggle.com/c/titanic/discussion/49105#279477), 
# I'll try two age bins ~ young and others
bins_age_2 = [ 0, 15, 100]
lab_ages_2 = [20,21]

for data_i in data:
    data_i['Age_bin_2'] = pd.cut(data_i.Age,bins_age_2,labels=lab_ages_2)
train = pd.concat([train, pd.get_dummies(train['Age_bin_2'], columns=['Age_bin_2'], prefix='Age_bin_2', prefix_sep='_')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Age_bin_2'], columns=['Age_bin_2'], prefix='Age_bin_2', prefix_sep='_')], axis=1)
features_to_use_2 = ['Pclass_1', 'Pclass_2', 'Pclass_3', 
                   'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S',
                   'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                   'Age_bin_2_0', 'Age_bin_2_1',
                   'Fare_pp', 'SibSp', 'Parch']
from sklearn.ensemble import RandomForestClassifier

random_search_rf2 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 5, train[features_to_use_2], y_all)
# random_search_rf2 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 100, train[features_to_use_2], y_all)
par =  {'max_depth': 12, 'criterion': 'gini', 'min_samples_leaf': 4, 'max_leaf_nodes': 17, 'min_samples_split': 2, 'n_estimators': 472,
       'n_jobs':1,
       'oob_score':True}
rf2 = train_cl_param(RandomForestClassifier, train[features_to_use_2], y_all, par)
#write_result(rf, 'random_forest_5.csv', test[features_to_use_2], pr_id)
get_importances(rf2, features_to_use_2)
# helper functions to generate distributions of hyper-parameters for NN

# learning rate of 0.0001--1, list of n_max floats 10**random_uniform
def rand_learning_rate(n_max=1000):
    return list(10.**np.random.uniform(-3,0,n_max))

# hidden layers: generates list of n_max tuples with 
# n_l_min--n_l_max integers, each between n_a_min and n_a_max
def rand_hidden_layer_sizes(n_l_min,n_l_max,n_a_min,n_a_max,n_max=1000):
    n_l = np.random.randint(n_l_min,n_l_max,n_max)
    list_hl = []
    for nl_i in n_l:
        list_hl.append(tuple(np.random.randint(n_a_min,n_a_max,nl_i)))
    return list_hl
# NN hyper parameters to test
param_dist_nn = {"activation": ["tanh", "relu"],
                 "learning_rate_init": rand_learning_rate(),
                 "hidden_layer_sizes": rand_hidden_layer_sizes(2,15,5,20),
                 "alpha": [0.00001,0.000006,0.000003,0.000001]
                }
from sklearn.neural_network import MLPClassifier

random_search_nn1 = model_random_search(MLPClassifier(batch_size=256), param_dist_nn, 5, X_all, y_all)
#random_search_nn1 = model_random_search(MLPClassifier(batch_size=256), param_dist_nn, 50, X_all, y_all)
par = {'learning_rate_init': 0.020871090748067426, 
       'alpha': 1e-06, 
       'activation': 'tanh', 
       'hidden_layer_sizes': (5, 9, 15, 10),
       'batch_size':256
       }
nn = train_cl_param(MLPClassifier, X_all, y_all, par)
#write_result(nn, 'nn_1.csv', X_test, pr_id)
