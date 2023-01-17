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

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 100
#FOR Kaggle

sample = pd.read_csv('/kaggle/input/mlbio1/sample_submission.csv')

test = pd.read_csv('/kaggle/input/mlbio1/test.csv')

train = pd.read_csv('/kaggle/input/mlbio1/train.csv')



#For local

# sample = pd.read_csv('healthcare-dataset-stroke-data/sample_submission.csv')

# test = pd.read_csv('healthcare-dataset-stroke-data/test.csv')

# train = pd.read_csv('healthcare-dataset-stroke-data/train.csv')
#Common Model Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from xgboost import XGBClassifier



#Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

#from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV



#Visualization

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

from sklearn.manifold import TSNE



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')



flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

pl = sns.color_palette(flatui)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
# As suggested in [], features with more than 15% missing data shold be dropped. That is why we drop smoking_status.

train = train.drop('smoking_status', axis=1)

test = test.drop('smoking_status', axis=1)
# As for bmi, we can either fill missing values with mean (or median) or drop rows with missing bmi data from the dataset.

# Here we will try filling with mean.

mean_bmi = train['bmi'].mean()

train['bmi'] = train['bmi'].fillna(mean_bmi)

test['bmi'] = test['bmi'].fillna(mean_bmi)
train.groupby('stroke').size().reset_index(name='counts')
global_mean_stroke = train['stroke'].mean()

global_mean_stroke
# gender

train[['gender', 'stroke']].groupby(['gender'], as_index=False).mean().sort_values(by='stroke', ascending=False)
train['gender'].value_counts()
fig, ax = plt.subplots(figsize=(8, 12), dpi=80)

plt.subplot(2,1,1)

sns.barplot(x='gender', y='stroke', data=train, palette=pl)

plt.subplot(2,1,2)

sns.countplot(x='gender', data=train, palette=pl)
# Conclusion: only 10 samples (compared to 30 000) have 'other' in gender column, thus it can be treated as n/a.

# We will use target encoding for gender, and will replace 'other' gender value with mean_stroke value

# (no need to even use smoothing with that few samples)

gender_target_dict = (train.groupby(['gender'])['stroke'].agg(['mean'])).to_dict()['mean']

gender_target_dict['Other'] = global_mean_stroke



train['gender_target_enc'] = train['gender'].replace(gender_target_dict)

test['gender_target_enc'] = test['gender'].replace(gender_target_dict)
# work_type

work_type_data = train[['work_type', 'stroke']].groupby(['work_type'], as_index=False).mean().sort_values(by='stroke', ascending=False)

work_type_data
train['work_type'].value_counts()
# coutplot for distribution of work_type and stroke probability plot

fig, ax = plt.subplots(figsize=(12, 12), dpi=80)

plt.subplot(2,1,1)

sns.barplot(x='work_type', y='stroke', data=train, palette=pl)

plt.subplot(2,1,2)

sns.countplot(x='work_type', data=train, palette=pl)



# We see, that 'Never_worked' appears to be rare, but not as much as 'other' gender,

# so here we will try using smoothed target encoding (https://maxhalford.github.io/blog/target-encoding-done-the-right-way/) and one-hot encoding



# smoothed target

n = train.groupby('work_type').size()

mean_by_work_type = train.groupby('work_type')['stroke'].mean()

m = 10 # smoothing coefficient

work_type_target_dict = ((mean_by_work_type*n + global_mean_stroke*m)/(n+m)).to_dict()



train['work_type_smoothed_target_enc'] = train['work_type'].replace(work_type_target_dict)

test['work_type_smoothed_target_enc'] = test['work_type'].replace(work_type_target_dict)

# one-hot

for work_type in train['work_type'].unique():

    train['work_type_is_{}'.format(work_type)] = (train['work_type'] == work_type)*1

    test['work_type_is_{}'.format(work_type)] = (test['work_type'] == work_type)*1
fig, ax = plt.subplots(figsize=(20, 8), dpi=80)

for i, binary_feature in enumerate(['ever_married', 'heart_disease', 'hypertension', 'Residence_type'], 1):

    # value counts

    print(train[binary_feature].value_counts())

    ax = plt.subplot(2, 4, i)

    # barplots

    sns.barplot(x=binary_feature, y='stroke', data=train, palette=pl)

    

    ax = plt.subplot(2, 4, 4 + i)

    # countplots

    sns.countplot(x=binary_feature, data=train, palette=pl)
# From the above barplot it can be seen that Residence_type variable is not important, thus we will not use it

train = train.drop('Residence_type', axis=1)

test = test.drop('Residence_type', axis=1)
# hypertension and heart_disease are already in binary incoding, so only ever_married is left to encode

train['ever_married'] = train['ever_married'].replace({'Yes':1, 'No':0 })

test['ever_married'] = test['ever_married'].replace({'Yes':1, 'No':0 })
# Comparing distributions of numeric features of samples grouped by target variable

fig, ax = plt.subplots(figsize=(10, 15), dpi=80)

for i, numeric_feature in enumerate(['age', 'bmi', 'avg_glucose_level'], 1):

    ax = plt.subplot(3, 1, i)

    sns.violinplot(y=numeric_feature, x='stroke', scale='area', data=train, palette=pl)
# age distributions vary the most, bmi - the least, though for now we will keep all 3 of numeric features
features = [ 'age', 'hypertension', 'heart_disease', 'ever_married',

        'avg_glucose_level',

        'gender_target_enc', 'work_type_smoothed_target_enc']

X_embedded = TSNE(n_components=2, random_state=21).fit_transform(train[features])
# stroke

plt.figure(figsize=(10,10))



plt.scatter(X_embedded[train['stroke']==0 ,0],

            X_embedded[train['stroke']==0 ,1],  c ="#3498db", s =1)





plt.scatter(X_embedded[train['stroke']==1 ,0],

            X_embedded[train['stroke']==1 ,1],  c ="#9b59b6", s =10)







plt.xlabel('t-sne 1')

plt.ylabel('t-sne 2')

plt.title('t-sne, stroke')

plt.show()
# color by every feature

features = features = [ 'age', 'hypertension', 'heart_disease', 'ever_married',

        'avg_glucose_level', 'bmi',

        'gender_target_enc', 'work_type_smoothed_target_enc']

plt.figure(figsize=(10,10*len(features)))



for i, feature in enumerate(features, 1):

    plt.subplot(len(features), 1, i)

    plt.scatter(X_embedded[... ,0],

                X_embedded[... ,1],  c =train[feature], s =2, cmap="Blues_r")

    plt.scatter(X_embedded[train['stroke']==1 ,0],

                X_embedded[train['stroke']==1 ,1],  c ="red", s=5)





    plt.xlabel('t-sne 1')

    plt.ylabel('t-sne 2')

    plt.title('t-sne ' + feature)
# tSNE plots confirm positive correlations with age, avg_glucose_level, heart_disease, hypertension and work_type

# correlations with bmi and gender are not seen that clearly
def your_cross_validation_for_roc_auc( clf, X, y ,cv=5):

    X = np.array(X.copy())

    y = np.array(y.copy())

    kf = KFold(n_splits=cv)

    kf.get_n_splits(X)

    scores = []

    for train_index, test_index in kf.split(X):

        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        clf.fit(X_train, y_train)

        prediction_on_this_fold = clf.predict_proba(X_test)[:,1]

        

        score = roc_auc_score(y_score=prediction_on_this_fold, y_true=y_test)

        scores.append(score)

        

    return scores
# Models

models = [

    SGDClassifier(loss='log', penalty = 'elasticnet', max_iter=50),

    XGBClassifier(n_estimators=45, max_depth=3),

    RandomForestClassifier(),

]

models_names = dict(zip(models, ['svc', 'xgb', 'gnb']))
all_features = [ 'age', 'hypertension', 'heart_disease', 'ever_married',

        'avg_glucose_level', 'bmi',

        'work_type_is_children',

        'work_type_is_Private', 'work_type_is_Never_worked',

        'work_type_is_Self-employed', 'work_type_is_Govt_job',

        'gender_target_enc', 'work_type_smoothed_target_enc']
# single feature scores

scores = {}



for model in models:

    features_scores = {}

    print(models_names[model])

    for f in all_features:

        scores = your_cross_validation_for_roc_auc(model, train[[f]] , train['stroke'])

        print(f, np.mean(scores))

        features_scores[f] = np.mean(scores)

    #print(models_names[model], features_scores)
features_target = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 

             'work_type_smoothed_target_enc', 'gender_target_enc']
features_onehot = [ 'age', 'hypertension', 'heart_disease', 'ever_married',

        'avg_glucose_level', 'bmi',

        'work_type_is_children',

        'work_type_is_Private', 'work_type_is_Never_worked',

        'work_type_is_Self-employed', 'work_type_is_Govt_job',

        'gender_target_enc']
models = [

    SGDClassifier(loss='log', penalty = 'elasticnet', max_iter=100),

    XGBClassifier(n_estimators=75, max_depth=3),

    RandomForestClassifier(n_estimators=75, max_depth=3),

]

models_names = dict(zip(models, ['svc', 'xgb', 'rfc']))

for model in models:

    print(models_names[model])

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    print(np.mean(scores_target), np.mean(scores_onehot))



# target enc best for svc, one-hot for random forest
# xgb

# n_estimators

x = []

y1 = []

y2 = []

for n_es in range(20,251,10):

    model = XGBClassifier(n_estimators=n_es, max_depth=3)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(n_es)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(n_es, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()
# xgb

# n_estimators (fine)

x = []

y1 = []

y2 = []

for n_es in range(60,81,1):

    model = XGBClassifier(n_estimators=n_es, max_depth=3)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(n_es)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(n_es, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()

# maximum at 75
# xgb

# max_depth

x = []

y1 = []

y2 = []

for m_d in range(1,10):

    model = XGBClassifier(n_estimators=75, max_depth=m_d)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(m_d)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(m_d, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()



# maximum at 3
# random forest

# n_estimators

x = []

y1 = []

y2 = []

for n_es in range(300,600,30):

    model = RandomForestClassifier(n_estimators=n_es, max_depth=7)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(n_es)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(n_es, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()



# maximum at ~530
# xgb

# max_depth

x = []

y1 = []

y2 = []

for m_d in range(1,10):

    model = RandomForestClassifier(n_estimators=530, max_depth=m_d)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(m_d)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(m_d, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()



# maximum at 7
# SGD

# max_iter

x = []

y1 = []

y2 = []

for n_es in range(2,400,10):

    model =  SGDClassifier(loss='log', penalty = 'elasticnet', max_iter=n_es)

    scores_target = your_cross_validation_for_roc_auc(model, train[features_target] , train['stroke'])

    scores_onehot = your_cross_validation_for_roc_auc(model, train[features_onehot] , train['stroke'])

    x.append(n_es)

    y1.append(np.mean(scores_target))

    y2.append(np.mean(scores_onehot))

    print(n_es, np.mean(scores_target), np.mean(scores_onehot))

fig, ax = plt.subplots(figsize=(10,8), dpi=80)

plt.plot(x, y1, label='target')

plt.plot(x, y2, label='onehot')

plt.show()



# plateau at ~100
# Best score was by xgboost on smoothed target encoding of worktype



best_model = XGBClassifier(n_estimators=75, max_depth=3)

best_model.fit(train[features_target], train['stroke'])

Y_pred = pd.DataFrame(best_model.predict_proba(test[features_target]))[1]

submission = pd.DataFrame({

        "id": test["id"],

        "stroke": Y_pred

    })

submission.to_csv("../working/submit.csv", index=False)

submission.sample(10)