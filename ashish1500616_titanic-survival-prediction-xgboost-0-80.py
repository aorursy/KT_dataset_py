# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import Libraries

import seaborn as sns

import matplotlib.pyplot as plt

import math

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

from mpl_toolkits.mplot3d import Axes3D

import missingno

import pylab 

import scipy.stats as stats

import lightgbm as lgb

import feature_engine

# Pretty display for notebooks

%matplotlib inline

plt.rcParams["figure.figsize"] = (20,10)

plt.style.use('seaborn-whitegrid')

import warnings

warnings.filterwarnings("ignore")





from sklearn.model_selection import train_test_split

import re

from feature_engine import missing_data_imputers as msi

from feature_engine import variable_transformers as vt

from feature_engine import outlier_removers as outr

from catboost import CatBoostClassifier, Pool, cv
data=pd.read_csv("../input/train.csv")

t_data=pd.read_csv("../input/test.csv")

data.head()
data.info()
# Describing all the Numerical Features

data.describe()
data.describe(include=['O'])


def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:

            g = sns.countplot(y=column, data=dataset)

            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            g = sns.distplot(dataset[column])

            plt.xticks(rotation=25)

    

plot_distribution(data.dropna(), cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
# How many missing values are there in our dataset?

missingno.matrix(data, figsize = (10,5))

missingno.bar(data, sort='ascending', figsize = (10,5))
vars_with_na=[]

def findVariablesWithMissingValues(df):

    # make a list of the variables that contain missing values

    global vars_with_na

    vars_with_na = [var for var in df.columns if df[var].isnull().sum()>1]



    # print the variable name and the percentage of missing values

    for var in vars_with_na:

        print(var, np.round(df[var].isnull().mean(), 3),  ' % missing values')



findVariablesWithMissingValues(data)
#  data[["Age","Survived"]

# mn=data.Age.mean()

# data.Age.fillna(mn)

# data['Age']=np.select([data["Age"]<=10,data["Age"]>35],["Child","Old"],default="Adult")

# data['Age']=np.select([data["Age"]<=10,data["Age"]>35,(data["Age"]>10) & (data["Age"]<=35)],["Child","Old","Adult"],default="NaN")

# data[["Age_","Age"]].head()

# data.drop("Age",axis=1,inplace=True)
l=['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Embarked']

def relBetVarSur(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(len(dataset)) / cols)

    for i, column in enumerate(dataset):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        data.groupby([column,'Survived'])["Sex"].count().plot.bar()

#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

#         g.set(yticklabels=substrings)

        plt.xticks(rotation=25)

    

relBetVarSur(l, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)
def analyse_na_value(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(len(dataset)) / cols)

    for i, column in enumerate(dataset):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        df = data.copy()

        df[column] = np.where(df[column].isnull(), 1, 0)

        df.groupby([column,'Survived'])["Sex"].count().plot.bar()

#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

#         g.set(yticklabels=substrings)

        plt.xticks(rotation=25)

    

analyse_na_value(vars_with_na, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)



# list of numerical variables

num_vars = [var for var in data.columns if data[var].dtypes != 'O']



print('Number of numerical variables: ', len(num_vars))



# visualise the numerical variables

data[num_vars].head()
#  list of discrete variables

discrete_vars = [var for var in num_vars if len(data[var].unique())<20 ]



print('Number of discrete variables: ', len(discrete_vars))
data[discrete_vars].head().drop('Survived',axis=1)
data.groupby(["Pclass"]).Survived.count()
def analyse_discrete(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(len(dataset)) / cols)

    for i, column in enumerate(dataset):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        df = data.copy()

        df.groupby([column]).Survived.count().plot.bar()

#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

#         g.set(yticklabels="Survived")

        ax.set_ylabel('No of Passengers')

        plt.xticks(rotation=25)

    

analyse_discrete(discrete_vars, cols=2, width=20, height=20, hspace=0.45, wspace=0.5)

# list of continuous variables

cont_vars = [var for var in num_vars if var not in discrete_vars]



print('Number of continuous variables: ', len(cont_vars))
data[cont_vars].head()
def analyse_continous(df, var):

    df = df.copy()

    plt.figure(figsize=(20,6))

    plt.subplot(1, 2, 1)

    df[var].hist(bins=20)

    plt.ylabel('Survived')

    plt.xlabel(var)

    plt.title(var)

    plt.subplot(1, 2, 2)

    stats.probplot(df[var], dist="norm", plot=pylab)

    plt.show()

    

    

for var in cont_vars[1:]:

    analyse_continous(data, var)
def find_outliers(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(len(dataset)) / cols)

    for i, column in enumerate(dataset):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        df = data.copy()

        df.boxplot(column=column)

#         substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

#         ax.set(yticklabels="Survived")

        ax.set_ylabel(column)

        plt.xticks(rotation=25)

    

find_outliers(cont_vars[1:], cols=2, width=20, height=20, hspace=0.45, wspace=0.5)





data.drop("Name",axis=1)





cat_vars = [var for var in data.columns if data[var].dtypes=='O']



print('Number of categorical variables: ', len(cat_vars))
data[cat_vars].head()


for var in cat_vars:

    print(var, len(data[var].unique()), ' categories')
features=data.drop(["Survived","Name","PassengerId","Ticket"],axis=1)

features_test=t_data.drop(["Name","PassengerId","Ticket"],axis=1)

labels=data.Survived
features_test.iloc[152]
features_test.info()


# X_train, X_test, y_train, y_test = train_test_split(features, labels,

#                                                     test_size=0.2,

#                                                     random_state=0) # we are setting the seed here

# X_train.shape, X_test.shape
features.Age.head()
median_imputer = msi.MeanMedianImputer(imputation_method='median', variables = ['Age',"Fare"])

median_imputer.fit(features)
median_imputer.imputer_dict_
features = median_imputer.transform(features)
features_test = median_imputer.transform(features_test)
plt.figure()

features.Age.plot(kind="kde")

plt.show()
findVariablesWithMissingValues(features)

# features = median_imputer.transform(features_test)

frequentLabel_imputer = msi.FrequentCategoryImputer(variables = ['Embarked'])

frequentLabel_imputer.fit(features)

features=frequentLabel_imputer.transform(features)



frequentLabel_imputer.imputer_dict_
features_test=frequentLabel_imputer.transform(features_test)
findVariablesWithMissingValues(features)
features.Embarked.value_counts().plot(kind="bar")
features.Cabin.isnull().sum()

def  findCabin(val):

    res=str()

    if val is not np.nan:

        match=re.findall("[A-Z]",str(val))

        for x in match:

            if x not in res:

                res+=x

        return res

    return np.nan

                



Cabin=features.Cabin.apply(findCabin)

Cabin.isnull().sum()
features.Cabin=Cabin

features.Cabin[128]
features_test.Cabin=features_test.Cabin.apply(findCabin)

features_test.head()


addLabel_imputer = msi.CategoricalVariableImputer(variables = ['Cabin'])

addLabel_imputer.fit(features)

features=addLabel_imputer.transform(features)

features_test=addLabel_imputer.transform(features_test)
findVariablesWithMissingValues(features)


features.Cabin.value_counts().plot(kind="bar")
features.info()
features.Age.min(),features.Age.max()
analyse_continous(features,"Age")
# bct = vt.BoxCoxTransformer(variables = ["Age"])

# bct.fit(features)

# features=bct.transform(features)
# windsoriser = outr.Windsorizer(distribution='gaussian', tail='right', fold=3, variables = ['Age'])

# windsoriser.fit(features)

# # windsoriser.right_tail_caps_

# features=windsoriser.transform(features)



def categAge(df):

    df['Age']=np.select([df["Age"].astype(int)<=10,df["Age"].astype(int)>35],["Child","Old"],default="Adult")
categAge(features)

features.head()
# # features_test=windsoriser.transform(features_test)

categAge(features_test)

features_test.head()
# analyse_continous(features,"Age")

# features.Age.min(),features.Age.max()
analyse_continous(features,"Fare")
features.Fare.min(),features.Fare.max()
# et = vt.ExponentialTransformer(variables = ['Fare'])

# et.fit(features)

# features=et.transform(features)
# analyse_continous(features,"Fare")
windsoriser = outr.Windsorizer(distribution='gaussian', tail='right', fold=3, variables = ['Fare'])

windsoriser.fit(features)

# windsoriser.right_tail_caps_

features=windsoriser.transform(features)
features_test=windsoriser.transform(features_test)
features.Fare.min(),features.Fare.max()
# for var in cont_vars[1:]:

#     find_outliers(features, var)
from feature_engine import categorical_encoders as ce
# features.columns
features.Pclass.value_counts()
features.Sex.value_counts()
# features.Sex=features.Sex.apply(lambda x: 1 if x=="male" else 0)#,features.Sex
# features_test.Sex=features_test.Sex.apply(lambda x: 1 if x=="male" else 0)#,features.Sex

# features_test.head()
features.Sex.value_counts()
features.head()
ohe_enc = ce.OneHotCategoricalEncoder(top_categories = None, variables = ["Age","Embarked","Sex","Cabin"], drop_last = False)

ohe_enc.fit(features)

features=ohe_enc.transform(features)

features_test=ohe_enc.transform(features_test)

ohe_enc.encoder_dict_
features.head()
features_test.head()
# count_enc = ce.CountFrequencyCategoricalEncoder(encoding_method = 'frequency',variables = [ 'Cabin'])

# count_enc.fit(features)

# features=count_enc.transform(features)

features.head()

# count_enc.fit(features_test)

# features_test=count_enc.transform(features_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(features)

scaler.transform(features_test)
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from xgboost import plot_tree

from xgboost import plot_importance

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

from numpy import sort

from skopt import BayesSearchCV


X_train, X_test, y_train, y_test = train_test_split(features, labels,

                                                    test_size=0.25,

                                                    random_state=0) # we are setting the seed here

X_train.shape, X_test.shape
# def status_print(optim_result):

#     """Status callback durring bayesian hyperparameter search"""

#     # Get all the models tested so far in DataFrame format

#     all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    

#     # Get current parameters and the best parameters    

#     best_params = pd.Series(bayes_cv_tuner.best_params_)

#     print('Model #{}\nBest Accuracy: {}\nBest params: {}\n'.format(

#         len(all_models),

#         np.round(bayes_cv_tuner.best_score_, 4),

#         bayes_cv_tuner.best_params_

#     ))

#     # Save all model results

#     clf_name = bayes_cv_tuner.estimator.__class__.__name__

#     all_models.to_csv(clf_name + "_cv_results.csv")


# bayes_cv_tuner = BayesSearchCV(estimator=xgb.XGBClassifier(

#     objective='binary:logistic',

#     eval_metric='logloss',

#     learning_rate=0.1,

#     n_estimators= 720,

#     sub_sample=0.76,

# #     early_stopping_rounds=106,

#     max_depth=3,

#     silent=0),

#     search_spaces={

# #     'learning_rate': (0.001, 0.1),

# #     'n_estimators': (0, 800),

# #     'sub_sample': (0.5, 1.0),

#     'early_stopping_rounds': (0, 1000),

# #     "max_depth":  (3, 25),

# #     "colsamplebylevel": (0.6, 1.0),

# #     'min_child_weight': [1, 7],

# #     'gamma': (0.05,1.0),

#     #         'min_child_weight': (15, 20),

# #         'max_depth': (6, 8),

# #         'alpha':(0,1.0)

#     #         'max_delta_step': (0, 20),

#     #         'subsample': (0.01, 1.0, 'uniform'),

#     #         'colsample_bytree': (0.01, 1.0, 'uniform'),

#     #         'colsample_bylevel': (0.01, 1.0, 'uniform'),

#     #         'reg_lambda': (1e-2, 1000, 'log-uniform'),

#     #         'reg_alpha': (1e-2, 1.0, 'log-uniform'),

#     #         'gamma': (1e-2, 0.5, 'log-uniform'),

#     #         'min_child_weight': (0, 20),

#     #         'scale_pos_weight': (1e-6, 500, 'log-uniform')

# },

# #     scoring='roc',

#     cv=StratifiedKFold(

#         n_splits=5,

#         shuffle=True,

#         random_state=0),

#     n_jobs=-1,

#     n_iter=20,

#     verbose=500,

#     refit=True,

#     random_state=0)

# result = bayes_cv_tuner.fit(

#     features.values, labels.values, callback=status_print)

bayes_cv_tuner.best_params_
# categorical_features_indices = np.where(features.dtypes != np.float)[0]

# model = CatBoostClassifier(

#     custom_loss=['Accuracy'],

#     random_seed=42,

#     logging_level='Silent'

# )

# model.fit(

#     X_train, y_train,

#     cat_features=categorical_features_indices,

#     eval_set=(X_test, y_test),

# #     logging_level='Verbose',  # you can uncomment this for text output

#     plot=True

# )
# cv_params = model.get_params()

# cv_params.update({

#     'loss_function': 'Logloss'

# })

# cv_data = cv(

#     Pool(features, labels, cat_features=categorical_features_indices),

#     cv_params,

#     plot=True

# )

# print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(

#     np.max(cv_data['test-Accuracy-mean']),

#     cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],

#     np.argmax(cv_data['test-Accuracy-mean'])

# ))

# model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')

# model_without_seed.fit(features, labels, cat_features=categorical_features_indices)



# print('Random seed assigned for this model: {}'.format(model_without_seed.random_seed_))


# params = {

#     'iterations': 500,

#     'learning_rate': 0.1,

#     'eval_metric': 'Accuracy',

#     'random_seed': 42,

#     'logging_level': 'Silent',

#     'use_best_model': False

# }

# train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

# validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)


# model = CatBoostClassifier(**params)

# model.fit(train_pool, eval_set=validate_pool)



# best_model_params = params.copy()

# best_model_params.update({

#     'use_best_model': True

# })

# best_model = CatBoostClassifier(**best_model_params)

# best_model.fit(train_pool, eval_set=validate_pool);



# print('Simple model validation accuracy: {:.4}'.format(

#     accuracy_score(y_test, model.predict(X_test))

# ))

# print('')



# print('Best model validation accuracy: {:.4}'.format(

#     accuracy_score(y_test, best_model.predict(X_test))

# ))
# model = CatBoostClassifier(

#     l2_leaf_reg=int(5.0),

#     learning_rate=0.1147638000846512,

#     iterations=500,

#     eval_metric='Accuracy',

#     random_seed=42,

#     logging_level='Silent'

# )

# cv_data = cv(Pool(features, labels, cat_features=categorical_features_indices), model.get_params())
# print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))
# xg_cl = xgb.XGBClassifier(objective='binary:logistic',learning_rate=0.01,subsample=0.55,n_estimators=200, seed=123)

xg_cl = xgb.XGBClassifier(objective='binary:logistic',

#     eval_metric='auc',

    learning_rate=0.1,

#     n_estimators= 720,

#     sub_sample=0.76,

#     max_depth=3

                         )

eval_set = [(X_train, y_train), (X_test, y_test)]

# eval_set = [(X_test, y_test)]

xg_cl.fit(X_train, y_train,eval_metric="auc",early_stopping_rounds=20, eval_set=eval_set, verbose=True)

plot_tree(xg_cl,num_trees=1, rankdir='LR')

plt.show()
# plot_importance(xg_cl)

# plt.show()
preds = xg_cl.predict(X_test)
accuracy = accuracy_score(y_test,preds)

accuracy
results = confusion_matrix(y_test, preds) 

print(results)
# dmatrix = xgb.DMatrix(data=features,label=labels)



# # params={"objective":"binary:logistic","max_depth":4}

# # tuned_params = {"objective":"binary:logistic",'colsample_bytree': 0.3, 'max_depth': 10,'subsample': 0.55, 'n_estimators': 200, 'learning_rate': 0.2}

# # tuned_params = {"objective":"binary:logistic",'learning_rate': 0.3}

# # tuned_params = {"objective":"binary:logistic","early_stopping_rounds":"6", "learning_rate":"0.08", "max_depth":"5", "n_estimators":"50"}

# tuned_params=bayes_cv_tuner.best_params_

# cv_results = xgb.cv(dtrain=dmatrix, params=tuned_params, nfold=5, num_boost_round=200, metrics="error",as_pandas=True, seed=123)



# # Print the accuracy



# print(((1-cv_results["test-error-mean"]).iloc[-1]))

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(xg_cl, features, labels, cv=kfold)

results.mean()
# results = xg_cl.evals_result()

# print(results)


results = xg_cl.evals_result()

epochs = len(results['validation_0']['auc'])

x_axis = range(0, epochs)

# plot log loss



# fig, ax = plt.subplots(figsize=(8,8))

# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

# ax.legend()

# plt.ylabel('Log Loss')

# plt.title('XGBoost Log Loss')

# plt.show()



# # plot classification error

# fig, ax = plt.subplots(figsize=(8,8))

# ax.plot(x_axis, results['validation_0']['error'], label='Train')

# ax.plot(x_axis, results['validation_1']['error'], label='Test')

# ax.legend()

# plt.ylabel('Classification Error')

# plt.title('XGBoost Classification Error')

# plt.show()



# plot ROC

fig, ax = plt.subplots(figsize=(12,12))

ax.plot(x_axis, results['validation_0']['auc'], label='Train')

ax.plot(x_axis, results['validation_1']['auc'], label='Test')

ax.legend()

plt.ylabel('Area Under Curve')

plt.title('XGBoost ROC')

plt.show()
# tuned_params = {"objective":"binary:logistic",'colsample_bytree': 0.3, 'max_depth': 10,'subsample': 0.55, 'n_estimators': 200, 'learning_rate': 0.2}

thresholds = sort(xg_cl.feature_importances_)

models = []

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(xg_cl, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model = xgb.XGBClassifier(objective='binary:logistic',

#     eval_metric='logloss',

    learning_rate=0.1,

#     n_estimators= 720,

#     sub_sample=0.76,

#     max_depth=3

                                       )

    selection_model.fit(select_X_train, y_train)

    # add model to models

    models.append([selection_model,selection])

    # eval model

    select_X_test = selection.transform(X_test)

    predictions = selection_model.predict(select_X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],

    accuracy*100.0))
# Finalize transformations

final_model = models[14][0]

final_selection = models[14][1]

final_X_train = final_selection.transform(X_train)



final_X_test = final_selection.transform(X_test)



final_y_pred = final_model.predict(final_X_test)

final_predictions = [round(value) for value in final_y_pred]



# Print evaluation metrics

accuracy = accuracy_score(y_test, final_predictions)

print("n=%d, Accuracy: %.2f%%" % (final_X_train.shape[1], accuracy*100.0))

confusion_matrix(y_test, final_predictions) 
kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(final_model

                          , features, labels, cv=kfold)

results.mean()
# selection = SelectFromModel(xg_cl, threshold=0.012, prefit=True)

# select_X_train = selection.transform(X_train)

# select_X_test = selection.transform(X_test)

# # another_model = xgb.XGBClassifier(objective='binary:logistic')

# another_model = xgb.XGBClassifier(objective='binary:logistic',**bayes_cv_tuner.best_params_)



# another_model.fit(select_X_train, y_train)



# select_y_pred = another_model.predict(select_X_test)

# select_predictions = [round(value) for value in select_y_pred]



# # Print evaluation metrics

# accuracy = accuracy_score(y_test, select_predictions)

# print("n=%d, Accuracy: %.2f%%" % (select_X_train.shape[1], accuracy*100.0))

# confusion_matrix(y_test, select_predictions) 
# features_test[features_test.Fare==np.nan]

# # features_test.head()

# features_test.isnull().all().all()

# features.isnull().all().all()
features_=final_selection.transform(features)

final_model.fit(features_,labels)

# final_model_2=xgb.XGBClassifier(objective='binary:logistic',**bayes_cv_tuner.best_params_)

# final_model_2.fit(features_,labels)

# another_model_2.fit(features_,labels)

# model.fit(features,labels, cat_features=categorical_features_indices)

# another_model_3.fit(features_,labels)
# features_


features_test_=final_selection.transform(features_test)

prediction = final_model.predict(features_test_)
# final_prediction=prediction+prediction_3

# final_prediction_=[int(x/2) for x in final_prediction]

# prediction_3=prediction_3.astype(int)



# final_prediction_=np.bitwise_and(prediction,prediction_3_b_opt)



final_prediction_=prediction
submission = pd.DataFrame({

        "PassengerId": t_data["PassengerId"],

        "Survived": final_prediction_



    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head(n=10)
len(submission[submission.Survived ==1 ])