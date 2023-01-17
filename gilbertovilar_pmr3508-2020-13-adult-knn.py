import os

import sys

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-darkgrid')

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col='Id', na_values="?")
train
rich = '>50K'

poor = '<=50K'



# Troca ponto por underline

link = {

    key: key.replace('.', '_')

    for key in train.columns

    

}



train = train.rename(columns=link)

train_rich = train[train['income']==f'{rich}']

train_poor = train[train['income']==f'{poor}']



#Encoder para transformar variavel categorica em numérica

link={

    '<=50K': 0,

    '>50K': 1

}

train['income_label'] = train['income'].apply(lambda x: link[x])



#função auxilar para exploração de dados

def explore(feature):

    explored_df_count = train[feature].value_counts().to_frame().reset_index().rename(columns={feature:'number of people', 'index':feature})

    explored_df_count['percent'] = explored_df_count['number of people']/len(train)*100

    explored_df_count = explored_df_count.set_index(feature).join(train_rich[feature].value_counts()).rename(columns={feature:f'{rich}'})

    explored_df_count['percent_rich'] = explored_df_count[f'{rich}']/explored_df_count['number of people']

    explored_df_count = explored_df_count.join(train_poor[feature].value_counts()).rename(columns={feature:f'{poor}'})

    explored_df_count['percent_poor'] = explored_df_count[f'{poor}']/explored_df_count['number of people']

    explored_df_count['percent_ratio'] = explored_df_count['percent_rich']/explored_df_count['percent_poor']

    explored_df_count = explored_df_count.fillna(0)

    return explored_df_count



## plotting settings

plt.rc('xtick',labelsize=15)

plt.rc('ytick',labelsize=15)

parameters = {

    'axes.titlesize': 25,

    'axes.labelsize': 20,

    'legend.fontsize': 20,

    'grid.alpha': 0.6

}

plt.rcParams.update(parameters)

train.info()
train.describe()
feature = 'age'

explored_df = explore(feature)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20,10))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

train_poor.hist(column=f'{feature}', ax=ax2, bins=len(explored_df), alpha=0.6)

train_rich.hist(column=f'{feature}', ax=ax2, bins=len(explored_df), alpha=0.6)



ax1.set_ylabel('number of people')

ax2.set_ylabel('number of people')

plt.tight_layout()

print(f'Global mean: {train[f"{feature}"].mean()}')

print(f'Low-end mean: {train_poor[f"{feature}"].mean()}')

print(f'High-end mean: {train_rich[f"{feature}"].mean()}')

#explored_df
feature = 'workclass'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['workclass', 'income']])

dummies.corr().iloc[:-2,-2:].sort_values(by='income_>50K')
feature = 'education'

explored_df = explore(feature)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

sns.boxplot(x='income', y='education_num', data=train[['education_num', 'income']], ax=ax2)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['education', 'income']])

dummies.corr().iloc[:-2,-2:].sort_values(by='income_>50K', ascending=False)
feature = 'marital_status'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['marital_status', 'income']])

dummies.corr().iloc[:-2,-2:].sort_values(by='income_>50K', ascending=False)
feature = 'occupation'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['occupation', 'income']])

dummies.corr().iloc[:-2,-2:].sort_values(by='income_>50K', ascending=False)
feature = 'relationship'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['relationship', 'income']])

dummies.corr().iloc[:-2,-2:].sort_values(by='income_>50K', ascending=False)
dummies = pd.get_dummies(train[['relationship', 'marital_status','income']])

fig, ax = plt.subplots(figsize=(15,10))

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

ax.set_yticklabels(ax.get_xticklabels(), fontsize=14)

sns.heatmap(data=dummies.corr(),ax=ax)

feature = 'race'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['race','income']])

fig, ax = plt.subplots(figsize=(10,7))

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

ax.set_yticklabels(ax.get_xticklabels(), fontsize=14)

sns.heatmap(data=dummies.corr(),ax=ax)

dummies.corr().iloc[:-2, -2:].sort_values(by='income_>50K')

feature = 'sex'

explored_df = explore(feature)

fig, ax1 = plt.subplots(1,1, figsize=(15,8))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

ax1.set_ylabel('number of people')

plt.tight_layout()

explored_df
dummies = pd.get_dummies(train[['sex','income']])

dummies.corr().iloc[:-2, -2:].sort_values(by='income_>50K')

feature = 'capital_gain'

explored_df = explore(feature)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(20,20))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

train_poor.hist(column=f'{feature}', ax=ax2, bins=10, alpha=0.6)

train_rich.hist(column=f'{feature}', ax=ax2, bins=10, alpha=0.6)

sns.boxplot(x='income', y=f'{feature}', data=train[[f'{feature}', 'income']], ax=ax3)

ax1.set_ylabel('number of people')

ax2.set_ylabel('number of people')

plt.tight_layout()

print(f'Global mean: {train[f"{feature}"].mean()}')

print(f'Low-end mean: {train_poor[f"{feature}"].mean()}')

print(f'High-end mean: {train_rich[f"{feature}"].mean()}')

#explored_df
feature = 'capital_loss'

explored_df = explore(feature)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(20,20))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

train_poor.hist(column=f'{feature}', ax=ax2, bins=10, alpha=0.6)

train_rich.hist(column=f'{feature}', ax=ax2, bins=10, alpha=0.6)

sns.boxplot(x='income', y=f'{feature}', data=train[[f'{feature}', 'income']], ax=ax3)

ax1.set_ylabel('number of people')

ax2.set_ylabel('number of people')

plt.tight_layout()

print(f'Global mean: {train[f"{feature}"].mean()}')

print(f'Low-end mean: {train_poor[f"{feature}"].mean()}')

print(f'High-end mean: {train_rich[f"{feature}"].mean()}')

#explored_df
plt.figure(figsize=(12, 6))

sns.heatmap(data=train[['capital_loss', 'capital_gain', 'income_label']].corr(), cmap='hot', linewidths=0.3, annot=True)
feature = 'hours_per_week'

explored_df = explore(feature)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(20,20))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

train_poor.hist(column=f'{feature}', ax=ax2, bins=30, alpha=0.6)

train_rich.hist(column=f'{feature}', ax=ax2, bins=30, alpha=0.6)

sns.boxplot(x='income', y=f'{feature}', data=train[[f'{feature}', 'income']], ax=ax3)

ax1.set_ylabel('number of people')

ax2.set_ylabel('number of people')

plt.tight_layout()

print(f'Global mean: {train[f"{feature}"].mean()}')

print(f'Low-end mean: {train_poor[f"{feature}"].mean()}')

print(f'High-end mean: {train_rich[f"{feature}"].mean()}')

#explored_df
feature = 'native_country'

explored_df = explore(feature)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,20))

explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

explored_df[explored_df.index!='United-States'][[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax2)

ax1.set_ylabel('number of people')

plt.tight_layout()

#explored_df
dummies = pd.get_dummies(train[['native_country','income']])

fig, ax = plt.subplots(figsize=(10,7))

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

ax.set_yticklabels(ax.get_xticklabels(), fontsize=14)

sns.heatmap(data=dummies.corr(),ax=ax)

#dummies.corr().iloc[:-2, -2:].sort_values(by='income_>50K')
feature = 'fnlwgt'

explored_df = explore(feature)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25,8))

#explored_df[[f'{poor}', f'{rich}']].sort_values(by=f'{feature}').plot.bar(stacked=True, alpha=0.6, ax=ax1)

train_poor.hist(column=f'{feature}', ax=ax1, bins=30, alpha=0.6)

train_rich.hist(column=f'{feature}', ax=ax1, bins=30, alpha=0.6)

sns.boxplot(x='income', y=f'{feature}', data=train[[f'{feature}', 'income']], ax=ax2)

plt.tight_layout()

print(f'Global mean: {train["fnlwgt"].mean()}')

print(f'Low-end mean: {train[train["income"]==f"{poor}"]["fnlwgt"].mean()}')

print(f'High-end mean: {train[train["income"]==f"{rich}"]["fnlwgt"].mean()}')
dummies = pd.get_dummies(train[['fnlwgt', 'income']])

dummies.corr()
plt.figure(figsize=(12, 6))

sns.heatmap(data=train.corr(), cmap='hot', linewidths=0.3, annot=True)
plt.figure(figsize=(12, 6))

dummies = pd.get_dummies(train[['marital_status', 'relationship']])

sns.heatmap(data=dummies.corr(), cmap='hot', linewidths=0.3, annot=True)
## Income também será excluída da base de treino e salva apenas como variável target (y)

drop_cols = ['education', 'native_country', 'fnlwgt', 'income', 'income_label', 'race', 'workclass']

y_train = train['income_label']

X_train = train.drop(columns=drop_cols)

X_train
X_train.isnull().sum()
from sklearn.impute import SimpleImputer



simple_imputer = SimpleImputer(strategy='most_frequent') #preenche com a moda

X_train[['occupation']] = simple_imputer.fit_transform(X_train[['occupation']])

X_train.isnull().sum()
from sklearn.preprocessing import RobustScaler



robust_scaler = RobustScaler()

X_train[['capital_gain', 'capital_loss']] = robust_scaler.fit_transform(X_train[['capital_gain', 'capital_loss']])

X_train = pd.get_dummies(X_train)
from sklearn.preprocessing import StandardScaler



stdscaler = StandardScaler()

X_train = stdscaler.fit_transform(X_train)
def build_classifier(X_train, y_train, X_test, y_test, clf_to_evaluate, score,param_grid, n_folds=5 ):

    print("# Tuning hyper-parameters for %s" % score)

    print()

    clf = GridSearchCV(clf_to_evaluate, param_grid, cv=n_folds, scoring=score, verbose=True, n_jobs=6, iid=False)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")

    print()

    print(clf.best_params_)

    print()

    print("Grid scores on development set:")

    print()

    means = clf.cv_results_['mean_test_score']

    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):

        print("%0.3f (+/-%0.03f) for %r" %(mean, std * 2, params))

    print()

    print("Detailed classification report:")

    print()

    y_true, y_pred = y_test, clf.predict(X_test)

    print(classification_report(y_true, y_pred))

    return clf, y_pred
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.1, random_state=42)



score = 'accuracy'

k_set = list(range(2,50))

params_grid = {'n_neighbors': k_set}



clf_to_evaluate = KNeighborsClassifier()

best_clf = build_classifier(X_train_split, y_train_split, X_test_split, y_test_split, clf_to_evaluate, score, params_grid)
plt.plot(k_set, best_clf[0].cv_results_['mean_test_score'])
knn = KNeighborsClassifier(**best_clf[0].best_params_)

knn.fit(X_train, y_train)
test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col='Id', na_values="?")

test.isnull().sum()
link = {

    key: key.replace('.', '_')

    for key in test.columns

    

}



test = test.rename(columns=link)



#drop_cols = ['education', 'native_country', 'fnlwgt', 'race']

drop_cols = ['education', 'native_country', 'fnlwgt', 'race', 'workclass']



# Feature selection

X_test = test.drop(columns=drop_cols)



# Missing data

# X_test[['occupation', 'workclass']] = simple_imputer.transform(X_test[['occupation', 'workclass']])

X_test[['occupation']] = simple_imputer.transform(X_test[['occupation']])



# Outliers

X_test[['capital_gain', 'capital_loss']] = robust_scaler.transform(X_test[['capital_gain', 'capital_loss']])



# Transformation

X_test = pd.get_dummies(X_test)



# Normalization

X_test = stdscaler.transform(X_test)

pred = knn.predict(X_test)

link = {

    1 : '>50K',

    0 : '<=50K'

}

pred = [link[value] for value in pred]

submission = pd.DataFrame()



submission[0] = test.index

submission[1] = pred

submission.columns = ['Id','income']

submission.to_csv('submission.csv', index = False)