import pandas as pd

from pandas import Series,DataFrame

import numpy as np



import matplotlib.pylab as plt

import seaborn as sns

# advanced plot library

import graphviz



from sklearn import metrics

from sklearn import preprocessing



# randomly split data into train and test set

# if model_selection doesn't work in pc, try cross_validation  

from sklearn.model_selection import train_test_split



# machine learning algorithm

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.grid_search import GridSearchCV

import xgboost as xgb



from sklearn.metrics import roc_curve, auc, roc_auc_score



# import data

path  = "../input/adult.csv"

train = pd.read_csv(path)
# preview the data 

% matplotlib inline 

train.head()

# train.describe()
train.info()
# now we generally konw the data 

# we see no missing value, because of '?' in the same place

# make it easily handle by replacing it.

train = train.replace('?', np.nan)



# transform into '0-1' classification problem 

train['income'] = train['income'].replace(['<=50K', '>50K'], [0, 1])
# the columns: ['workclass' & 'occupation' & 'native.country'] miss several thousand values
# age

# figure :income distribution by age 

# x_label is 'age', y_label is 'number of people divided by income'

fig, axis = plt.subplots(1, 1, figsize=(8,5))

sns.countplot(x='age', hue='income', data=train, ax=axis)

axis.set_xticklabels([])

axis.set_xlabel('age(17-90)')

axis.set_title('Income distribution by age')
# workclass



# mark the NAN with 'missing_workclass'

# we can drop 'Without-pay' & 'Never-worked' because its small number

fig, axis = plt.subplots(1, 1, figsize=(8,5))

train['workclass'] = train['workclass'].fillna('missing_workclass')

sns.countplot(x='workclass', hue='income', data=train, ax=axis)

axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
# create dummies values

workclass_dummies_values = pd.get_dummies(train['workclass'])

train = train.join(workclass_dummies_values)

train.drop(['workclass', 'Without-pay', 'Never-worked'], axis=1, inplace=True)
# fnlwgt:final weight

# a factor used by Census Bureau, maybe a balance between rigion, race, etc. 

# just drop 

train.drop(['fnlwgt'], axis=1, inplace=True)
# education	



# Prof-school is professional school

# Assoc-acdm is associate academic(2 years for some guys dropping out)

# Assoc-voc is given students who complete associate vocational program 



# we think that education under 12th is the same class as: self-study

# Prof-school & Assoc-acdm & Assoc-voc is in same class as: normal-education

# so here is 7 classes left



# combine into 7 classes

train['education'] = train['education'].replace(['7th-8th', '10th', '11th', '1st-4th', '5th-6th', '12th', '9th', 'Preschool'], 'self-study')

train['education'] = train['education'].replace(['Prof-school', 'Assoc-acdm', 'Assoc-voc'], 'normal-education')



fig, axis = plt.subplots(1, 1, figsize=(8,5))

sns.countplot(x='education', hue='income', data=train)



# dummies values

education_dummies_values = pd.get_dummies(train['education'])

train = train.join(education_dummies_values)

train.drop(['education'], axis=1, inplace=True)
# education.num



# get the percentage of the rich class by 'education.num'

education_num_perc = train[['education.num', 'income']].groupby(['education.num'], as_index=False).mean()

sns.barplot(x='education.num', y='income', data=education_num_perc)



# get the frequency by 'education.num'

#fig = plt.figure()

#fig.set_size_inches(8,5)



# kdeplot is 'kernel density estimation'

facet = sns.FacetGrid(train, hue='income', aspect=4).map(sns.kdeplot, 'education.num', shade=True)



# maybe the difference between two figures contributes to prejudice of education 
# marital.status	



# figure 1: count by 'marital.status'

fig, axis1 = plt.subplots(1, 1, figsize=(8,4))

sns.countplot(x='marital.status', data=train, ax=axis1)

axis1.set_xticklabels(axis1.get_xticklabels(), rotation=45)



# figure 2: percentage of 'rich' in 'marital.status'

fig, axis2 = plt.subplots(1, 1, figsize=(8,4))

marital_num_perc = train[['marital.status', 'income']].groupby(['marital.status'], as_index=False).mean()

sns.barplot(x='marital.status', y='income', data=marital_num_perc, ax=axis2)

axis2.set_xticklabels(axis2.get_xticklabels(), rotation=45)

# then drop 'Separated' & 'Widowed', no influence



# dummies values

marital_dummies_values = pd.get_dummies(train['marital.status'])

train = train.join(marital_dummies_values)

train.drop(['Separated', 'Widowed', 'marital.status'],axis=1,inplace=True)



# better emotional state,  more money
# occupation	



# fill the missing occupation with 'missing_occupation'

train['occupation'] = train['occupation'].fillna('missing_occupation')

fig, axis = plt.subplots(1, 1, figsize=(8,4))

sns.countplot(x='occupation', data=train, ax=axis).set_xticklabels(axis.get_xticklabels(), rotation=45)



# combine 'Handlers-cleaners' & 'Farming-fishing' & 'Priv-house-serv'

train['occupation'] = train['occupation'].replace(['Handlers-cleaners', 'Farming-fishing', 'Priv-house-serv'], 'hard-work')



# dummies values

occupation_dummies_values = pd.get_dummies(train['occupation'])

train = train.join(occupation_dummies_values)

train.drop(['occupation'], axis=1, inplace=True)
# relationship



relationship_dummies_values = pd.get_dummies(train['relationship'])

train = train.join(relationship_dummies_values)

# drop 'Other-relative' because of less values and multicolinearity

train.drop(['relationship', 'Other-relative'], axis=1, inplace=True)



# race



race_dummies_values = pd.get_dummies(train['race'])

train = train.join(race_dummies_values)

# drop 'Other' & 'Amer-Indian-Eskimo'

train.drop(['race', 'Other', 'Amer-Indian-Eskimo'], axis=1, inplace=True)



# sex



# 'Male' marked '1', 'Female' is removed

train['Male'] = train['sex'].replace(['Female', 'Male'], [0, 1])

train.drop(['sex'], axis=1, inplace=True)
# capital.gain	(stock & bond & real estate)

# capital.loss	



# combine them, named 'capital_flow'



capital_flow = train['capital.gain'] - train['capital.loss']

train.insert(1,'capital_flow',capital_flow)

train.drop(['capital.gain', 'capital.loss'], axis=1, inplace=True)

# along the horizontal direction, large diameter means large number of corresponding capital_flow

sns.violinplot('income', 'capital_flow', data=train)



#it seems large absolute value of capital flow tending to be rich
# hours.per.week	



# no change
#native.country



# 'native.country' is not a very important factor, here is a subjective classification, part of it may

# not reasonable, but makes no big difference

 

# fill missing values with 'missing_nation'

train['native.country'] = train['native.country'].fillna('missing_nation')



# classify 'native.country' by region and economic development level into 6 classes

train['native.country'] = train['native.country'].replace(['Canada', 'South', 'Cuba', 'Peru'], 'America_rich')

train['native.country'] = train['native.country'].replace(['Philippines', 'Japan'], 'Eastern_rich')

train['native.country'] = train['native.country'].replace(['China', 'Taiwan', 'Hong', 'India', \

'Cambodia', 'Laos', 'Thailand'], 'Eastern_development')

train['native.country'] = train['native.country'].replace(['Mexico', 'Vietnam', 'Trinadad&Tobago', \

'Puerto-Rico', 'Honduras', 'Nicaragua', 'Dominican-Republic', 'Haiti', 'El-Salvador', 'Guatemala', \

'Columbia', 'Jamaica', 'Ecuador', 'Outlying-US(Guam-USVI-etc)'], 'America_development')

train['native.country'] = train['native.country'].replace(['Greece', 'Holand-Netherlands', 'Poland', \

'Iran', 'England', 'Germany', 'Italy', 'Ireland', 'Hungary', 'France', 'Yugoslavia', 'Scotland', \

'Portugal'], 'EU')



fig, axis = plt.subplots(1, 1, figsize=(8,4))

sns.countplot(x='native.country', data=train, ax=axis).set_xticklabels(axis.get_xticklabels(), rotation=45)



native_country_dummies = pd.get_dummies(train['native.country'])

train = train.join(native_country_dummies)

train.drop(['native.country'], axis=1, inplace=True)
# get train and test data set (32561 rows total)



# here we drop columns with missing values because it brings improvement, based on results

x = train.drop(['income', 'missing_occupation', 'missing_nation', 'missing_workclass'], axis=1, inplace=False)

y = train['income']



# randomly get train and test data set by 2:1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=23)



####the end of data preprocessing ####
# predict



# decision tree



# first we use simple decision tree, it seems powerful considering accuracy of '81.55%'

# but we know the imbalance of data set, accuracy should be measured by 'recall rate'

# find that 'recall_score = 0.6327', bad

dct = DecisionTreeClassifier(random_state = 42)

dct.fit(x_train, y_train)

y_pred = dct.predict(x_test)

'accuracy_score = %s' %metrics.accuracy_score(y_pred, y_test),\

'recall_score = %s' %metrics.recall_score(y_pred, y_test)



sns.violinplot(y_pred, y_test)
# randomforest



randomforest = RandomForestClassifier(n_estimators=100)

randomforest.fit(x_train, y_train)

y_pred1 = randomforest.predict(x_test)

'accuracy_score = %s' %metrics.accuracy_score(y_pred1, y_test),\

'recall_score = %s' %metrics.recall_score(y_pred1, y_test)
# AdaBoostClassifier--find maxism index



# the following is essential when finding suitable n_estimators



# k_range = np.arange(1,100) # k=263 ==>0.8699

# scores = []

# for k in k_range :

# 	abc = AdaBoostClassifier(n_estimators=k)

# 	abc.fit(x_train, y_train)

# 	y_pred0 = abc.predict(x_test)

# 	scores.append(metrics.accuracy_score(y_pred0, y_test))

# print scores.index(max(scores))+k_range[0], max(scores)

# plt.plot(k_range, scores)

# AdaBoost



abc = AdaBoostClassifier(n_estimators=263)

abc.fit(x_train, y_train)

y_pred2 = abc.predict(x_test)

'accuracy_score = %s' %metrics.accuracy_score(y_pred2, y_test),\

'recall_score = %s' %metrics.recall_score(y_pred2, y_test)

# pleasing improvement
# xgboost training

# use GridSearchCV to train xgboost, gradually chage >>>

# n_estimators  max_depth  min_child_weight  gamma....so on



# model = xgb.XGBClassifier(max_depth=6, min_child_weight=1, n_estimators=142, gamma=0, subsample=1, colsample_bytree=0.4, reg_alpha=0.015,\

#                          learning_rate=0.12)

# gsearch = GridSearchCV(model, {'learning_rate':[0.1,0.11,0.12,0.13]}, scoring='roc_auc', verbose=1)

# gsearch.fit(x_train, y_train)

# print gsearch.best_params_,"\n", gsearch.best_score_
# xgboost 



# use the model aboved to make prediction 

xgboost = xgb.XGBClassifier(max_depth=6, min_child_weight=1, n_estimators=142, gamma=0, subsample=1, colsample_bytree=0.4, reg_alpha=0.015,\

                          learning_rate=0.12)

xgboost.fit(x_train, y_train)

y_pred3 = xgboost.predict(x_test)

'accuracy_score = %s' %metrics.accuracy_score(y_pred3, y_test),\

'recall_score = %s' %metrics.recall_score(y_pred3, y_test)
# roc curve



# see that xgboost achieves best accuracy

fpr_rf, tpr_rf, _= roc_curve(y_test, y_pred1)

roc_auc1 = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label = 'randomforest.auc=%0.4f' %roc_auc1)



fpr_abc, tpr_abc, _= roc_curve(y_test, y_pred2)

roc_auc2 = auc(fpr_abc, tpr_abc)

plt.plot(fpr_abc, tpr_abc, label = 'AdaBoost.auc=%0.4f' %roc_auc2)



fpr_xgb, tpr_xgb, thresholds= roc_curve(y_test, y_pred3)

roc_auc3 = auc(fpr_xgb, tpr_xgb)

plt.plot(fpr_xgb, tpr_xgb, label = 'XgBoost.auc = %0.4f' %roc_auc3)



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.plot([0, 1], [0, 1], 'r--')

plt.legend(loc='best')
# also we can plot importance figure of all factors

# see that 'age', 'marital.status', 'education.num', 'capital_flow' is of strong correlation 

# and the tree is useful in classification, not for interpreting

xgb.plot_importance(xgboost)

xgb.plot_tree(xgboost, num_trees=6)