# Get Data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import random as rn

import os



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC, LinearSVC

from keras import backend as K

import tensorflow as tf

from keras.models import Sequential, Model

from keras.layers import Dense, BatchNormalization, Activation, Input, Dropout

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

from keras import initializers



%matplotlib inline
np.random.seed(42)

rn.seed(42)

tf.random.set_seed(42)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
train.shape, test.shape
train_test = pd.concat([train, test], axis = 0)
train_test.head()
train_test.shape
train_test["PassengerId"].describe()
train_test["PassengerId"].value_counts().sort_values()
train_test["Survived"].describe()
train_test["Survived"].value_counts().sort_values()/train.shape[0]
train["Pclass"].describe()
train["Pclass"].value_counts().sort_values()/train.shape[0]
test["Pclass"].describe()
test["Pclass"].value_counts().sort_values()/test.shape[0]
sum(train_test["Name"][train_test["Name"].isnull()])
train_test["Name"].value_counts().sort_values()
train_test[train_test["Name"] == "Kelly, Mr. James"]
train_test[train_test["Name"] == "Connolly, Miss. Kate"]
sum(train_test["Sex"].isnull())
train["Sex"].value_counts().sort_values()/train.shape[0]
test["Sex"].value_counts().sort_values()/test.shape[0]
train["Age"].describe()
print(sum(train["Age"].isnull()))

sum(train["Age"].isnull())/train.shape[0]
test["Age"].describe()
print(sum(test["Age"].isnull()))

sum(test["Age"].isnull())/test.shape[0]
train["SibSp"].describe()
train["SibSp"].value_counts().sort_values()
train["SibSp"].value_counts().sort_values()/train.shape[0]
test["SibSp"].describe()
test["SibSp"].value_counts().sort_values()
test["SibSp"].value_counts().sort_values()/test.shape[0]
train["Parch"].describe()
train["Parch"].value_counts().sort_values()
train["Parch"].value_counts().sort_values()/train.shape[0]
test["Parch"].describe()
test["Parch"].value_counts().sort_values()
test["Parch"].value_counts().sort_values()/test.shape[0]
train_test["Ticket"].value_counts().sort_values()
sum(train_test["Ticket"].isnull())
train["Fare"].describe()
train["Fare"].value_counts().sort_values()
test["Fare"].describe()
test["Fare"].value_counts().sort_values()
sum(test["Fare"].isnull())
sum(train["Cabin"].isnull())/train.shape[0]
train["Cabin"].value_counts().sort_values()
sum(test["Cabin"].isnull())/test.shape[0]
test["Cabin"].value_counts().sort_values()
train["Embarked"].value_counts().sort_values()/train.shape[0]
sum(train["Embarked"].isnull())
test["Embarked"].value_counts().sort_values()/test.shape[0]
sum(test["Embarked"].isnull())
for dataset in [train, test, train_test]:

    dataset["Alone_YN"] = (dataset["SibSp"] == 0) & (dataset["Parch"] == 0)
train["Ticket"][train["Embarked"].isnull()]
train[train["Ticket"] == "113572"]
train["Embarked"][train["Ticket"] == "113572"] = "S"

train_test["Embarked"][train_test["Ticket"] == "113572"] = "S"
train[train["Ticket"] == "113572"]
# Too many missing values, just drop it

train.drop(["Cabin"], axis = 1, inplace = True)

test.drop(["Cabin"], axis = 1, inplace = True)

train_test.drop(["Cabin"], axis = 1, inplace = True)
train_test
test[test["Fare"].isnull()]
train_test["Fare"].median(), test["Fare"].median(), train["Fare"].median()
# impute

test["Fare"][test["Fare"].isnull()] = train_test["Fare"].median()

train_test["Fare"][train_test["Fare"].isnull()] = train_test["Fare"].median()
train_test.info()
# extract title from name

train["Title"] = train["Name"].map(lambda x: re.search(", [^\.]+\.", x)[0][2:-1])

test["Title"] = test["Name"].map(lambda x: re.search(", [^\.]+\.", x)[0][2:-1])

train_test["Title"] = train_test["Name"].map(lambda x: re.search(", [^\.]+\.", x)[0][2:-1])
train_test["Title"].unique()
test.groupby(["Title"]).PassengerId.count().sort_values()
train.groupby(["Title"]).PassengerId.count().sort_values()
train.groupby(["Title"]).Survived.mean().sort_values()
# review data of passengers with titles other than: Mr, Mrs, Miss, Master

train_test[["PassengerId", "Pclass", "Sex", "Age", "Title"]][train_test["Title"].isin([x for x in train_test["Title"].unique() if x not in ['Mr', 'Mrs', 'Miss', 'Master']])]
# check median age of male doctor passengers

train_test["Age"][(train_test["Title"] == "Dr") & (train_test["Sex"] == "male")].median()
# impute missing age of male Dr (PassengerId: 767)

train_test["Age"][train_test["PassengerId"] == 767] = 47

train["Age"][train["PassengerId"] == 767] = 47
# group titles into categories: Mr, Ms, Miss, Master



def title_chage(dataset, index):

    

    passenger = dataset.iloc[index]



    if passenger["Sex"] == "male":



        if passenger["Age"] < 18:

            

            return "Master"

    

        else:

            

            return "Mr"

        

    else:

        

        if passenger["Age"] < 18:

            

            return "Miss"

        

        else:

            

            return "Ms"
for dataset in [train, test, train_test]:

    dataset["Title"] = [title_chage(dataset, i) for i in range(len(dataset))]
train_test["Title"].unique(), train["Title"].unique(), test["Title"].unique()
# visualise Age distributions by Titles "Mr" and "Ms"



for title in ["Mr", "Ms"]:



    sns.set(rc={'figure.figsize':(40,10)})

    sns.set(font_scale=2)



    # get data

    age_dist = train_test["Age"][(train_test["Title"] == title) & (train_test["Age"].notnull())]

    

    # Boxplot

    plt.subplot(1, 3, 1)  

    g = sns.boxplot(age_dist, orient='v')

    g.text(0.5, 1, title + ": Age Dist - Box Plot", size=24, ha="center", transform=g.transAxes)

    g.ylab = 'Age'



    # Histogram

    plt.subplot(1, 3, 2)  

    g = sns.distplot(age_dist, kde=False)

    g.text(0.5, 1, title + ": Age Dist - Histogram", size=24, ha="center", transform=g.transAxes)

    g.ylab = 'Frequency'

              

    # Histogram

    plt.subplot(1, 3, 3)  

    g = sns.kdeplot(age_dist, cumulative=True)

    g.text(0.5, 1, title + ": Age Dist - Cum Freq", size=24, ha="center", transform=g.transAxes)

    g.ylab = 'Cum Freq'



    plt.show()
# impute Age in Train dataset using the exponential distribution

num_imputes_needed = len(train[train["Age"].isnull()])

a = np.floor(np.random.exponential(scale=18, size=num_imputes_needed))

g = sns.distplot(a, kde=False)
train["Age"][train["Age"].isnull()] = a

print(len(train["Age"][train["Age"] > 80]))

train["Age"][train["Age"] > 80] = 80. # since only one such datapoint
# impute Age in Test dataset using the exponential distribution

num_imputes_needed = len(test[test["Age"].isnull()])

a = np.floor(np.random.exponential(scale=18, size=num_imputes_needed))

g = sns.distplot(a, kde=False)
test["Age"][test["Age"].isnull()] = a

print(len(test["Age"][test["Age"] > 80]))

test["Age"][test["Age"] > 80] = 80. # since only two such datapoints
train_test = pd.concat([train, test], axis = 0)
train_test.info()
for dataset in [train, test, train_test]:

    dataset["NameLen"] = dataset["Name"].apply(lambda x: len(x))
test_PIDs = test.pop("PassengerId")



test.drop(["Name", "Ticket"], axis = 1, inplace = True)



for dataset in [train, train_test]:

    

    dataset.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)
y_train_test = train_test.pop("Survived")

y_train = train.pop("Survived")
train.info()
# convert Pclass into categorical

for dataset in [train, test, train_test]:

    dataset["Pclass"] = dataset["Pclass"].apply(lambda x: str(x))
# get dummies

train = pd.get_dummies(train, drop_first = True)

test = pd.get_dummies(test, drop_first = True)

train_test = pd.get_dummies(train_test, drop_first = True)
# convert Alone_YN into int

for dataset in [train, test, train_test]:

    dataset["Alone_YN"] = [1 if x else 0 for x in dataset["Alone_YN"]]
# Get Train Test Splits

np.random.seed(0)

train, val, y_train, y_val = train_test_split(train, y_train, train_size = 0.7, test_size = 0.3, random_state = 100)
train.describe()
scaler = MinMaxScaler()



train[["Age", "SibSp", "Parch", "Fare", "NameLen"]] = scaler.fit_transform(train[["Age", "SibSp", "Parch", "Fare", "NameLen"]])

val[["Age", "SibSp", "Parch", "Fare", "NameLen"]] = scaler.transform(val[["Age", "SibSp", "Parch", "Fare", "NameLen"]])

test[["Age", "SibSp", "Parch", "Fare", "NameLen"]] = scaler.transform(test[["Age", "SibSp", "Parch", "Fare", "NameLen"]])
test.head()
sns.set(font_scale=1)

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, linecolor='white', annot=True)
# Titles highly correlated with sex; at least drio Title_Mr



for dataset in [train, val, test, train_test]:

    dataset.drop(["Title_Mr"], axis = 1, inplace = True)
plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, linecolor='white', annot=True)
# retain a copy of the datasets for logistic regression

train_logistic = train.copy()

val_logistic = val.copy()

test_logistic = test.copy()



# train model

logistic_1 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_1 = logistic_1.fit()

print(logistic_1.summary())
# remove Embarked_Q due to high p-value

for dataset in [train_logistic, val_logistic, test_logistic]:

    dataset.drop(["Fare"], axis = 1, inplace = True)



# re-train model

logistic_2 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_2 = logistic_2.fit()

print(logistic_2.summary())
# remove Embarked_Q due to high p-value

for dataset in [train_logistic, val_logistic, test_logistic]:

    dataset.drop(["Embarked_Q"], axis = 1, inplace = True)



# re-train model

logistic_3 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_3 = logistic_3.fit()

print(logistic_3.summary())
# remove Embarked_S due to high p-value

for dataset in [train_logistic, val_logistic, test_logistic]:

    dataset.drop(["Embarked_S"], axis = 1, inplace = True)



# re-train model

logistic_4 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_4 = logistic_4.fit()

print(logistic_4.summary())
# remove NameLen due to high p-value

for dataset in [train_logistic, val_logistic, test_logistic]:

    dataset.drop(["NameLen"], axis = 1, inplace = True)



# re-train model

logistic_5 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_5 = logistic_5.fit()

print(logistic_5.summary())
# Check VIF

vif = pd.DataFrame()

vif['Features'] = train_logistic.columns

vif['VIF'] = [np.round(variance_inflation_factor(train_logistic.values, i), 2) for i in range(train_logistic.shape[1])]

vif = vif.sort_values(by = "VIF", ascending = False)



vif
# remove Sex_male due to high VIF

for dataset in [train_logistic, val_logistic, test_logistic]:

    dataset.drop(["Sex_male"], axis = 1, inplace = True)



# re-train model

logistic_6 = sm.GLM(y_train, sm.add_constant(train_logistic), family = sm.families.Binomial())

logistic_6 = logistic_6.fit()

print(logistic_6.summary())
# Check VIF

vif = pd.DataFrame()

vif['Features'] = train_logistic.columns

vif['VIF'] = [np.round(variance_inflation_factor(train_logistic.values, i), 2) for i in range(train_logistic.shape[1])]

vif = vif.sort_values(by = "VIF", ascending = False)



vif
# Predict

y_val_pred_prob = logistic_6.predict(sm.add_constant(val_logistic))
# Find optimal cutoff point for converting churn probabilities to churn predictions



opt_cut_off = pd.DataFrame(columns = ['Cutoff','Acc']) # create empty dataframe



for i in np.arange(0, 1.1, 0.1): # for each cutoff point, get precision, recall

    y_val_pred = list(map(lambda x: 1 if x > i else 0, y_val_pred_prob))

    opt_cut_off = opt_cut_off.append({'Cutoff': i, 'Acc': metrics.accuracy_score(y_val, y_val_pred)}, ignore_index = True)



# Plot precision-recall curve

opt_cut_off.plot(x='Cutoff', y = ['Acc'])
# Considering cutoff to be 0.6, predict Survival

y_val_pred_logml = list(map(lambda x: 1 if x > 0.6 else 0, y_val_pred_prob))



# Compute final accuracy, precision and recall scores

print(metrics.accuracy_score(y_val, y_val_pred_logml))

print(metrics.classification_report(y_val, y_val_pred_logml))

print(metrics.confusion_matrix(y_val, y_val_pred_logml))
# Considering cutoff to be 0.6, predict Survival

y_train_pred_prob = logistic_6.predict(sm.add_constant(train_logistic))

y_train_pred_logml = list(map(lambda x: 1 if x > 0.6 else 0, y_train_pred_prob))



# Compute final accuracy, precision and recall scores

print(metrics.accuracy_score(y_train, y_train_pred_logml))

print(metrics.classification_report(y_train, y_train_pred_logml))

print(metrics.confusion_matrix(y_train, y_train_pred_logml))
# validation and training sets are performing comparably, so no over fitting



# but others in the competition have 100% accuracy, so model is under-fitting the data, so try other non-linear models
# Instantiate Model with Default Hyperparameters

rfc = RandomForestClassifier(oob_score=True, random_state=42)



# Train Model

rfc.fit(train, y_train)



# Predict

y_train_pred = rfc.predict(train)

y_val_pred = rfc.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising max_depth

param_grid = {'max_depth': range(1, 25, 1)}



# Instantiate grid search

rfc_grid_search = RandomForestClassifier(random_state=42)

rfc_grid_search = GridSearchCV(estimator = rfc_grid_search, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

rfc_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rfc_grid_search.best_score_, rfc_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rfc_grid_search.cv_results_["param_max_depth"], 

         rfc_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rfc_grid_search.cv_results_["param_max_depth"], 

         rfc_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rfc_grid_search.best_estimator_.predict(train)

y_val_pred = rfc_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# tuning other hyper-parameters did not help

# validation and training sets are performing comparably, so not too much over fitting

# random forest not improving accuracy as compared to log reg probably because there aren't very many features

# but others in the competition have 100% accuracy, so model is under-fitting the data, so try other models
# Convert Data to DMatrix

X_train_xgb = xgb.DMatrix(train, label=y_train)

X_val_xgb = xgb.DMatrix(val)



# Train Model with default hyper-parameters

bst = xgb.train({'objective':'binary:logistic'}, X_train_xgb)



# Predict

y_train_pred = list(map(lambda x: 1 if x > 0.5 else 0, bst.predict(X_train_xgb)))

y_val_pred = list(map(lambda x: 1 if x > 0.5 else 0, bst.predict(X_val_xgb)))



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising eta

param_grid = {'eta': [10e-4, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}



# Instantiate grid search

xgb_search = XGBClassifier(random_state=42)

xgb_grid_search = GridSearchCV(estimator = xgb_search, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

xgb_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(xgb_grid_search.best_score_, xgb_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(xgb_grid_search.cv_results_["param_eta"], 

         xgb_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(xgb_grid_search.cv_results_["param_eta"], 

         xgb_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("eta")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = xgb_grid_search.best_estimator_.predict(train)

y_val_pred = xgb_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising min_child_weight

param_grid = {'eta': [0.001],

              'min_child_weight': [0.1, 1, 5, 10]}



# Instantiate grid search

xgb_search = XGBClassifier(random_state=42)

xgb_grid_search = GridSearchCV(estimator = xgb_search, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

xgb_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(xgb_grid_search.best_score_, xgb_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(xgb_grid_search.cv_results_["param_min_child_weight"], 

         xgb_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(xgb_grid_search.cv_results_["param_min_child_weight"], 

         xgb_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("min_child_weight")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = xgb_grid_search.best_estimator_.predict(train)

y_val_pred = xgb_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising min_child_weight

param_grid = {'eta': [0.001],

              'min_child_weight': [5],

              'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]}



# Instantiate grid search

xgb_search = XGBClassifier(random_state=42)

xgb_grid_search = GridSearchCV(estimator = xgb_search, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

xgb_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(xgb_grid_search.best_score_, xgb_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(xgb_grid_search.cv_results_["param_gamma"], 

         xgb_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(xgb_grid_search.cv_results_["param_gamma"], 

         xgb_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("gamma")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = xgb_grid_search.best_estimator_.predict(train)

y_val_pred = xgb_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# tuning other hyper-parameters did not help

# there is a little over fitting

# xgboost also not improving accuracy as compared to log reg probably because there aren't very many features

# but others in the competition have 100% accuracy, so model is not performing well, so try other models
# Instantiate Model with Default Hyperparameters

lsvc = LinearSVC(random_state=42)



# Train Model

lsvc.fit(train, y_train)



# Predict

y_train_pred = lsvc.predict(train)

y_val_pred = lsvc.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# tuning hyper-parameter C did not really help

# significant over-fitting not evident

# Linear SVM not improving accuracy as compared to log reg probably because underlying relationship may be non-linear
# Instantiate Model with Default Hyperparameters

rbfsvc = SVC(kernel = 'rbf', random_state=42)



# Train Model

rbfsvc.fit(train, y_train)



# Predict

y_train_pred = rbfsvc.predict(train)

y_val_pred = rbfsvc.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Improve fit via hyperparameter tuning

param_grid = {

    'C': [0.01, 0.1, 1, 10, 100, 1000],

    'kernel': ['rbf']

}



# Instantiate grid search

rbfsvc = SVC(random_state=42)

rbfsvc_grid_search = GridSearchCV(estimator = rbfsvc, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='recall', return_train_score=True)



# Run grid search on dataset

rbfsvc_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rbfsvc_grid_search.best_score_, rbfsvc_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rbfsvc_grid_search.cv_results_["param_C"], 

         rbfsvc_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rbfsvc_grid_search.cv_results_["param_C"], 

         rbfsvc_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("C")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rbfsvc_grid_search.best_estimator_.predict(train)

y_val_pred = rbfsvc_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Improve fit via hyperparameter tuning

param_grid = {

    'C': [0.1],

    'kernel': ['rbf'],

    'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1]

}



# Instantiate grid search

rbfsvc = SVC(random_state=42)

rbfsvc_grid_search = GridSearchCV(estimator = rbfsvc, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='recall', return_train_score=True)



# Run grid search on dataset

rbfsvc_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rbfsvc_grid_search.best_score_, rbfsvc_grid_search.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rbfsvc_grid_search.cv_results_["param_gamma"], 

         rbfsvc_grid_search.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rbfsvc_grid_search.cv_results_["param_gamma"], 

         rbfsvc_grid_search.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("gamma")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rbfsvc_grid_search.best_estimator_.predict(train)

y_val_pred = rbfsvc_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# model fairly invariant to hyper-parameter tuning

# little or no over-fitting evident

# RBF SVM not improving accuracy; try poly SVM
# Instantiate Model with Default Hyperparameters

polsvc = SVC(kernel = 'poly', random_state=42)



# Train Model

polsvc.fit(train, y_train)



# Predict

y_train_pred = polsvc.predict(train)

y_val_pred = polsvc.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Specify universe of potential values for hyperparameters 

param_grid = {

    'C': [0.01, 0.1, 0.5, 1, 10, 100, 1000],

    'kernel': ['poly'],

    'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1],

    'degree': [2, 3, 4, 5, 6],

    'max_iter': [1000000]

}



# Instantiate grid search

polsvc = SVC(random_state=42)

polsvc_grid_search = GridSearchCV(estimator = polsvc, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='recall', return_train_score=True)



# Run grid search on dataset

polsvc_grid_search.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(polsvc_grid_search.best_score_, polsvc_grid_search.best_params_))



# Use best model to predict

y_train_pred = polsvc_grid_search.best_estimator_.predict(train)

y_val_pred = polsvc_grid_search.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# no improvement via hyperparameter tuning

# little or no over-fitting evident

# Poly SVM not improving accuracy; try Neural Networks
## Clear prev sessions

K.clear_session()



## instantiate

nn_model = Sequential()



## layer 1



nn_model.add(Dense(8, input_dim=13, activation='relu'))



## layer 2



nn_model.add(Dense(16, activation = 'relu'))



## layer 3 - output



nn_model.add(Dense(1, activation='sigmoid'))



## model summary



nn_model.summary()
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(x=train, y=y_train, batch_size=16, epochs=20, verbose=1, validation_data=(val, y_val), shuffle=True, initial_epoch=0)
y_train_pred = np.array([1 if x > 0.5 else 0 for x in nn_model.predict(train, batch_size=16)])

y_val_pred = np.array([1 if x > 0.5 else 0 for x in nn_model.predict(val, batch_size=16)])
sum(y_train_pred == y_train)/y_train.shape[0], sum(y_val_pred == y_val)/y_val.shape[0]
# Bias problem -- try increasing number of layers in NN and N per layer



## Clear prev sessions

K.clear_session()



## instantiate

nn_model1 = Sequential()



## layer 1



nn_model1.add(Dense(32, input_dim=13, activation='relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 2



nn_model1.add(Dense(64, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 3



nn_model1.add(Dense(64, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 4



nn_model1.add(Dense(128, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 5



nn_model1.add(Dense(64, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 6



nn_model1.add(Dense(64, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 7



nn_model1.add(Dense(32, activation = 'relu'))

nn_model1.add(Dropout(0.3))

nn_model1.add(BatchNormalization())



## layer 8 - output



nn_model1.add(Dense(1, activation='sigmoid'))



## model summary



nn_model1.summary()
nn_model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model1.fit(x=train, y=y_train, batch_size=16, epochs=100, verbose=1, validation_data=(val, y_val), shuffle=True, initial_epoch=0)
y_train_pred = np.array([1 if x > 0.5 else 0 for x in nn_model1.predict(train, batch_size=16)])

y_val_pred = np.array([1 if x > 0.5 else 0 for x in nn_model1.predict(val, batch_size=16)])
sum(y_train_pred == y_train)/y_train.shape[0], sum(y_val_pred == y_val)/y_val.shape[0]
# Build gen function to predict Survived given algorithm name, potential hyperparameter values etc using CV grid search

def predict_and_incorporate(algorithm, params, datasets, num_folds, metric):

    

    p_i_grid_search = GridSearchCV(estimator = algorithm, param_grid = params, cv = num_folds,

                           n_jobs = -1, verbose = 0, scoring=metric)

    

    p_i_train, p_i_y_train, p_i_val, p_i_y_val, p_i_test = datasets

    

    # Run grid search on dataset

    p_i_grid_search = p_i_grid_search.fit(p_i_train, p_i_y_train)

    

    # Predict

    p_i_y_train_pred = p_i_grid_search.best_estimator_.predict(p_i_train)

    p_i_y_val_pred = p_i_grid_search.best_estimator_.predict(p_i_val)    

    

    # Evaluate

    print('Training Accuracy is {}'.format(metrics.accuracy_score(p_i_y_train, p_i_y_train_pred)))

    print('Validation Accuracy is {}'.format(metrics.accuracy_score(p_i_y_val, p_i_y_val_pred)))

    

    # Send back predictions for augmenting train / val / test datasets

    # with predictions from the given algorithm as a new feature [Ensemble strategy]

    

    return p_i_y_train_pred, p_i_y_val_pred, p_i_grid_search.best_estimator_.predict(p_i_test)
# Set up parameters for the various algorithms to be tried out

random_forest_params = {

    'max_depth': [5],

    'random_state': [42]

}

xgboost_params = {

    'eta': [0.001],

    'min_child_weight': [5],

    'gamma': [5],

    'random_state': [42]

}

lsvm_params = {'random_state': [42]}
# Specify folds

num_folds = KFold(n_splits=3, shuffle=True, random_state=42)
# Instantiate algorithms

rfc = RandomForestClassifier()

xgbc = XGBClassifier()

lsvc = LinearSVC()
# Copy Datasets prior to Augmentation

train_copy = train.copy()

val_copy = val.copy()

test_copy = test.copy()



# Build list of datasets for input into grid search

datasets = [train_copy, y_train, val_copy, y_val, test_copy]



# List Models (Algorithms), Model Names (Algo Names) and Params

algo_names_params = [(rfc, "Random_Forest", random_forest_params), (xgbc, "XGBoost", xgboost_params), (lsvc, "Lin_SVC", lsvm_params)]
# Augment Dataset

for algorithm, algo_name, params in algo_names_params:

    train[algo_name], val[algo_name], test[algo_name] = predict_and_incorporate(algorithm, params, \

                                                           datasets, num_folds, metric = "accuracy")
# Add on predictions from logistic regression as well



train["Logistic"] = logistic_6.predict(sm.add_constant(train_logistic))

val["Logistic"] = logistic_6.predict(sm.add_constant(val_logistic))

test["Logistic"] = logistic_6.predict(sm.add_constant(test_logistic))
# Add on predictions from neural networks as well



train["NN"] = nn_model1.predict(train_copy, batch_size=16)

val["NN"] = nn_model1.predict(val_copy, batch_size=16)

test["NN"] = nn_model1.predict(test_copy, batch_size=16)
train.describe()
sns.set(font_scale=1)

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, linecolor='white', annot=True)
# Instantiate

rfc_ensemble = RandomForestClassifier()



# Fit

rfc_ensemble.fit(train, y_train)



# Predict

y_train_pred = rfc_ensemble.predict(train)

y_val_pred = rfc_ensemble.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising max_depth

param_grid = {'max_depth': range(1, 25, 1)}



# Instantiate grid search

rfc_grid_search_final = RandomForestClassifier(random_state=42)

rfc_grid_search_final = GridSearchCV(estimator = rfc_grid_search_final, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy',return_train_score=True)



# Run grid search on dataset

rfc_grid_search_final.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rfc_grid_search_final.best_score_, rfc_grid_search_final.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rfc_grid_search_final.cv_results_["param_max_depth"], 

         rfc_grid_search_final.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rfc_grid_search_final.cv_results_["param_max_depth"], 

         rfc_grid_search_final.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rfc_grid_search_final.best_estimator_.predict(train)

y_val_pred = rfc_grid_search_final.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising n_estimators

param_grid = {

    'max_depth': [2],

    'n_estimators': [10, 20, 30, 40, 50, 75, 100, 150]

}



# Instantiate grid search

rfc_grid_search_final = RandomForestClassifier(random_state=42)

rfc_grid_search_final = GridSearchCV(estimator = rfc_grid_search_final, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

rfc_grid_search_final.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rfc_grid_search_final.best_score_, rfc_grid_search_final.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rfc_grid_search_final.cv_results_["param_n_estimators"], 

         rfc_grid_search_final.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rfc_grid_search_final.cv_results_["param_n_estimators"], 

         rfc_grid_search_final.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("n_estimators")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rfc_grid_search_final.best_estimator_.predict(train)

y_val_pred = rfc_grid_search_final.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
# Regularise over-fitting by optimising max_features

param_grid = {

    'max_depth': [2],

    'n_estimators': [100],

    'max_features': range(1, train.shape[1], 1)

}



# Instantiate grid search

rfc_grid_search_final = RandomForestClassifier(random_state=42)

rfc_grid_search_final = GridSearchCV(estimator = rfc_grid_search_final, param_grid = param_grid, cv = 3,

                           n_jobs = -1, verbose = 10, scoring='accuracy', return_train_score=True)



# Run grid search on dataset

rfc_grid_search_final.fit(train, y_train)



# Best stats

print("Best stats {}, {}".format(rfc_grid_search_final.best_score_, rfc_grid_search_final.best_params_))



# plot accuracy vs. parameter curve

plt.figure()

plt.plot(rfc_grid_search_final.cv_results_["param_max_features"], 

         rfc_grid_search_final.cv_results_["mean_train_score"], 

         label="training acc")

plt.plot(rfc_grid_search_final.cv_results_["param_max_features"], 

         rfc_grid_search_final.cv_results_["mean_test_score"], 

         label="test acc")

plt.xlabel("max_features")

plt.ylabel("accuracy")

plt.legend()

plt.show()



# Use best model to predict

y_train_pred = rfc_grid_search_final.best_estimator_.predict(train)

y_val_pred = rfc_grid_search_final.best_estimator_.predict(val)



# Evaluate

print('Training Accuracy is {}'.format(metrics.accuracy_score(y_train, y_train_pred)))

print('Validation Accuracy is {}'.format(metrics.accuracy_score(y_val, y_val_pred)))
y_test_prob = [1 if x > 0.5 else 0 for x in rfc_grid_search_final.best_estimator_.predict(test)]



output = pd.DataFrame({'PassengerId': test_PIDs, 'Survived': y_test_prob})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")