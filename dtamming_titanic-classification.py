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
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

matplotlib.style.use('fivethirtyeight')



from scipy import stats

import statsmodels.api as sm



from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, KFold, cross_validate, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from xgboost import XGBClassifier



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
folder = '/kaggle/input/titanic/'

train = pd.read_csv(folder+'train.csv')

test = pd.read_csv(folder+'test.csv')

print(train.shape)

print(test.shape)
train_ids = train['PassengerId'].values

test_ids = test['PassengerId'].values

data = pd.concat([train, test]).set_index('PassengerId')
data.info()
sns.heatmap(data[['Survived', 'Age', 'Fare']].corr(), annot=True);
pd.crosstab(data['Survived'], data['Sex'])
stats.chi2_contingency(pd.crosstab(data['Survived'], data['Sex']))[:2]
ticket_counts = data['Ticket'].value_counts()

data['SharedTicketWith'] = data['Ticket'].apply(lambda x: ticket_counts[x] - 1)

sns.countplot(data['SharedTicketWith']);
data['FamSize'] = data['SibSp'] + data['Parch']
sns.heatmap(data[['Survived', 'SharedTicketWith', 'FamSize', 'SibSp', 'Parch']].corr(method='spearman'), annot=True);
data['Cabin'].value_counts(dropna=False)
cabin_counts = data['Cabin'].value_counts()

data['CabinOccupants'] = data['Cabin'].apply(lambda cabin: 0 if isinstance(cabin, float) and np.isnan(cabin) else cabin_counts[cabin])

sns.countplot(x = 'CabinOccupants', hue='Survived', data=data);
names = data['Name']

# sort to ensure longest possible title is chosen

title_list = sorted(['Mr', 'Mrs', 'Ms', 'Miss', 'Master', 'Rev', 'Dr', 'Col', 'Major', 'Capt', 'Jonkheer', 'Countess', 'Mme', 'Don', 'Mlle', 'Dona'], key=len, reverse=True)

names[names.apply(lambda s : not any(t in s for t in title_list))]
titles = names.apply(lambda s : [t for t in title_list if t in s][0])

titles.value_counts()
titles = titles.replace(to_replace=['Ms', 'Mlle'], value='Miss')

titles = titles.replace({'Mme': 'Mrs'})

titles = titles.replace(to_replace=['Col', 'Major', 'Capt'], value='Military')

titles = titles.replace(to_replace=['Dona', 'Don', 'Countess', 'Jonkheer', 'Dr', 'Rev'], value='Rare')

titles.value_counts()
fig, ax = plt.subplots(figsize=(16,6));

graph = sns.countplot(x=titles, hue=data['Survived']);

graph.set_xticklabels(graph.get_xticklabels(), rotation=45);
pd.crosstab(data['Survived'], titles)
data['Title'] = titles
grid = sns.FacetGrid(row='Sex', hue='Survived', data=data, aspect=4, height=4);

grid.map(sns.countplot, 'SharedTicketWith');

grid.add_legend();
grid = sns.FacetGrid(row='Sex', hue='Survived', data=data, aspect=4, height=4);

grid.map(sns.countplot, 'FamSize');

grid.add_legend();
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))

sns.countplot(x='Embarked', hue='Survived', data=data, ax=ax[0,0]);

sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax[0,1]);

sns.countplot(x='Sex', hue='Survived', data=data, ax=ax[0,2])

sns.kdeplot(data[data['Survived'] == 0]['Age'], shade=True, color='blue', label=0, ax=ax[1,0]);

sns.kdeplot(data[data['Survived'] == 1]['Age'], shade=True, color='red', label=1, ax=ax[1,0]);

sns.kdeplot(data[data['Survived'] == 0]['Fare'], shade=True, color='blue', label=0, ax=ax[1,1]);

sns.kdeplot(data[data['Survived'] == 1]['Fare'], shade=True, color='red', label=1, ax=ax[1,1]);

sns.countplot(x='CabinOccupants', hue='Survived', data=data, ax=ax[1,2]);
print(data['Fare'].skew())

print(data['Age'].skew())
sns.kdeplot(np.log1p(data[data['Survived'] == 0]['Fare']), shade=True, color='blue', label=0);

sns.kdeplot(np.log1p(data[data['Survived'] == 1]['Fare']), shade=True, color='red', label=1);
print(np.log1p(data['Fare']).skew())
data.info()
data[data['Embarked'].isna()]
h = sns.FacetGrid(data, col='Pclass');

h.map(sns.countplot, 'Embarked');

h.add_legend();
data.loc[[62, 830], 'Embarked'] = 'Q'
data[data['Fare'].isna()]
grid = sns.FacetGrid(data=data, hue='Pclass', height=4, aspect=4);

grid.map(sns.kdeplot, 'Fare', shade=True);

grid.add_legend();
data.loc[1044, 'Fare'] = data[data['Pclass'] == 3]['Fare'].median()
sns.boxplot(x='Title', y='Age', data=data);

data[data['Age'].isna()]['Title'].value_counts()
age_by_title = data.groupby('Title')['Age'].median()

data['Age'] = data.apply(lambda row : age_by_title[row['Title']] if pd.isna(row['Age']) else row['Age'], axis=1)
data = data.drop(columns=['Ticket', 'SibSp', 'Parch', 'Cabin', 'Name'])
data['Fare_log1p'] = np.log1p(data['Fare'])

data = data.drop(columns=['Fare'])
data['IsMale'] = data['Sex'].apply(lambda x: x == 'male')

data = data.drop(columns=['Sex'])
data['Pclass'] = data['Pclass'].astype(str)

data['Survived'] = data['Survived'].astype(bool)
data.info()
data.to_csv('data_nodummies.csv')
data = pd.read_csv('data_nodummies.csv', index_col='PassengerId')
train = data.loc[train_ids]

test = data.loc[test_ids]
train = pd.get_dummies(train, drop_first=True)

test = pd.get_dummies(test, drop_first=True)
test = test.drop(columns=['Survived'])

X = train.drop(columns=['Survived'])

y = train['Survived']
best_params = {}
def search_1d(model, scaler, param_name, param_list, n_repeats=1, xscale='linear'):

    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)

    grid = GridSearchCV(model, {param_name: param_list}, cv=kfold)

    if scaler is not None:

        pipe = make_pipeline(scaler, grid)

        pipe.fit(X, y)

        grid = pipe[1]

    else:

        grid.fit(X, y)

    accs = np.array(grid.cv_results_['mean_test_score'])

    graph = sns.lineplot(param_list, accs)

    graph.set(xlabel=param_name, ylabel='Accuracy', xscale=xscale)

    return grid.best_params_
best_params['knn'] = search_1d(KNeighborsClassifier(), MinMaxScaler(), 'n_neighbors', list(range(3, 15)), n_repeats=10)
best_params['svc_linear'] = search_1d(SVC(kernel='linear'), RobustScaler(), 'C', [10**i for i in range(0, 3)], n_repeats=1, xscale='log')
best_params['svc_linear'] = search_1d(SVC(kernel='linear'), RobustScaler(), 'C', [10 + i for i in range(-5, 6)], n_repeats=1)
search_1d(SVC(), RobustScaler(), 'C', [10**i for i in range(-1, 2)], n_repeats=10, xscale='log')
search_1d(SVC(), RobustScaler(), 'gamma', [10**i for i in range(-2, 1)], n_repeats=10, xscale='log')
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

params = {'C': [0.5, 0.8, 1, 3, 6], 

          'gamma': [0.01, 0.02]}

pipe = make_pipeline(

    RobustScaler(), GridSearchCV(SVC(), params, cv=kfold)

)

pipe.fit(X, y)



accs = np.array(pipe['gridsearchcv'].cv_results_['mean_test_score'])

best_params['svc_rbf'] = pipe['gridsearchcv'].best_params_

print(best_params['svc_rbf'])

print(pipe['gridsearchcv'].best_score_)
search_1d(DecisionTreeClassifier(), None, 'max_depth', [2**i for i in range(5)], n_repeats=10)
search_1d(DecisionTreeClassifier(), None, 'min_samples_leaf', [2**i for i in range(5)], n_repeats=10)
search_1d(DecisionTreeClassifier(), None, 'min_samples_split', [2**i for i in range(1, 7)], n_repeats=10)
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

params = {'max_depth': [8, 9, 10],

         'min_samples_leaf': [5, 6, 7],

         'min_samples_split': [10, 13, 16, 19, 22]}

grid = GridSearchCV(DecisionTreeClassifier(), params, cv=kfold)

grid.fit(X, y)

best_params['tree'] = grid.best_params_

print(best_params['tree'])

print(grid.best_score_)
search_1d(RandomForestClassifier(), None, 'n_estimators', list(range(60,150,10)), n_repeats=10)
search_1d(RandomForestClassifier(n_estimators=60), None, 'max_depth', [4, 6, 8, 10, 12, 14, 16], n_repeats=10)
search_1d(RandomForestClassifier(max_depth=8), None, 'n_estimators', list(range(60,100,10)), n_repeats=10)
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

params = {'n_estimators': [70, 80],

          'max_depth': [8, 9]}

grid = GridSearchCV(RandomForestClassifier(), params, cv=kfold)

grid.fit(X, y)

best_params['forest'] = grid.best_params_

print(best_params['forest'])

print(grid.best_score_)
search_1d(XGBClassifier(), None, 'learning_rate', [0.15, 0.2, 0.25], n_repeats=10)
fixed_params = {'learning_rate': 0.2}

search_1d(XGBClassifier(**fixed_params), None, 'n_estimators', list(range(90, 120, 10)), n_repeats=10)
fixed_params['n_estimators'] = 100

search_1d(XGBClassifier(**fixed_params),

          None, 'max_depth', list(range(2, 9)), n_repeats=10)
search_1d(XGBClassifier(**fixed_params),

         None, 'min_child_weight', list(range(4, 10)), n_repeats=10)
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

params = {'max_depth': [3, 4, 5, 6, 7],

         'min_child_weight': [7, 8, 9, 10]}

grid = GridSearchCV(XGBClassifier(**fixed_params), params, cv=kfold)

grid.fit(X, y)

print(grid.best_params_)

print(grid.best_score_)
fixed_params['max_depth'] = 7

fixed_params['min_child_weight'] = 8

search_1d(XGBClassifier(**fixed_params),

         None, 'gamma', [0, 0.1, 0.2], n_repeats=10)
fixed_params['gamma'] = 0
search_1d(XGBClassifier(**fixed_params),

         None, 'subsample', [0.1*i for i in range(8, 11)], n_repeats=10)
search_1d(XGBClassifier(**fixed_params),

         None, 'colsample_bytree', [0.1*i for i in range(5, 11)], n_repeats=10)
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

params = {'subsample': [0.9, 1],

         'colsample_bytree': [0.9, 1]}

grid = GridSearchCV(XGBClassifier(**fixed_params), params, cv=kfold)

grid.fit(X, y)

print(grid.best_params_)

print(grid.best_score_)
fixed_params['subsample'] = 1

fixed_params['colsample_bytree'] = 1
del fixed_params['learning_rate']
search_1d(XGBClassifier(**fixed_params),

         None, 'learning_rate', [0.05, 0.1, 0.15, 0.2], n_repeats=10)
fixed_params['learning_rate'] = 0.1

del fixed_params['n_estimators']
search_1d(XGBClassifier(**fixed_params),

         None, 'n_estimators', list(range(80, 150, 10)), n_repeats=10)
fixed_params['n_estimators'] = 100

best_params['xgb'] = fixed_params
best_params
def score_model(model, n_repeats, metric_list, include_std=False):

    metrics_dict = {metric : [] for metric in metric_list}

    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)

    results = cross_validate(model, X, y, scoring=metric_list, cv=kfold)

    for metric in metric_list:

        metrics_dict[metric].append(results['test_'+metric].mean())

    if include_std:

        return {metric : (np.mean(vec), np.std(vec)) for metric, vec in metrics_dict.items()}

    else:

        return {metric : np.mean(vec) for metric, vec in metrics_dict.items()}
classifiers = {

    'logistic_regression': make_pipeline(RobustScaler(), LogisticRegression()),

    'knn': make_pipeline(MinMaxScaler(), KNeighborsClassifier(**best_params['knn'])),

    'svc_linear': make_pipeline(RobustScaler(), SVC(kernel='linear', **best_params['svc_linear'])),

    'svc_rbf': make_pipeline(RobustScaler(), SVC(**best_params['svc_rbf'])),

    'tree': DecisionTreeClassifier(**best_params['tree']),

    'forest': RandomForestClassifier(**best_params['forest']),

    'xgb': XGBClassifier(**best_params['xgb'])

}
metric_list = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']

val_accs_fine = pd.DataFrame(columns=metric_list)

for name, model in classifiers.items():

    val_accs_fine.loc[name] = score_model(model, 10, metric_list)

val_accs_fine
classifiers = {

    'logistic_regression': make_pipeline(RobustScaler(), LogisticRegression()),

    'knn': make_pipeline(MinMaxScaler(), KNeighborsClassifier()),

    'svc_linear': make_pipeline(RobustScaler(), SVC(kernel='linear')),

    'svc_rbf': make_pipeline(RobustScaler(), SVC()),

    'tree': DecisionTreeClassifier(),

    'forest': RandomForestClassifier(),

    'xgb': XGBClassifier()

}
metric_list = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']

val_accs_def = pd.DataFrame(columns=metric_list)

for name, model in classifiers.items():

    val_accs_def.loc[name] = score_model(model, 10, metric_list)

val_accs_def
fig, ax = plt.subplots(figsize=(20,10))

sns.scatterplot(x=val_accs_fine.index, y=val_accs_fine['accuracy'], label='Fine-Tuned', ax=ax, s=100)

sns.scatterplot(x=val_accs_def.index, y=val_accs_def['accuracy'], label='Default', ax=ax, s=100)

plt.legend();
xgb = XGBClassifier(**best_params['xgb']).fit(X,y)

preds = xgb.predict(test).astype(int)
preds_df = pd.DataFrame({'PassengerId': test.index, 'Survived':preds})

preds_df.head()
preds_df.to_csv('submission.csv', index=False)