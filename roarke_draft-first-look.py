# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train/train.csv')
df_train.head()
df_train.describe()
#Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
sns.set(style="darkgrid")
ax = sns.countplot(x="Gender", data=df_train)
#Type - Type of animal (1 = Dog, 2 = Cat)
#We'll add type of animal to the plot above
ax = sns.countplot(x="Type", hue="Gender", data=df_train)
g = sns.FacetGrid(df_train, row="Type", col="Gender", margin_titles=True)
bins = np.linspace(0, 36, 12)
g.map(plt.hist, "Age", color="steelblue", bins=bins)
df_train[df_train['Type'] == 1]['Age'].describe()
df_train[df_train['Type'] == 2]['Age'].describe()
sum(df_train['Breed2'] == 0) / len(df_train)
#ax = sns.countplot(x="Breed1", data=df_train)
df_train_by_breed = df_train.groupby('Breed1').agg(['count', 'mean'])
df_train_by_breed.sort_values(by=[('Type', 'count')], ascending=False)
ax = sns.countplot(x="AdoptionSpeed", hue="Gender", data=df_train)
Y = df_train['AdoptionSpeed'].values
X = df_train.drop(['AdoptionSpeed', 'Name', 'Description', 'PetID', 'RescuerID'], axis=1).values

np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]
cutoff = int(0.75*len(X))

train_data, train_labels = X[:cutoff], Y[:cutoff]
test_data, test_labels = X[cutoff:], Y[cutoff:]
# some decision tree and random forest imports
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=0, max_depth=5)
dt.fit(train_data, train_labels)
y_preds = dt.predict(test_data)
## Let's define the kappa scoring metric for use in our evaluations
def metric(y1,y2):
    return cohen_kappa_score(y1,y2, weights='quadratic')

# Make scorer for scikit-learn
scorer = make_scorer(metric)
metric(y_preds, test_labels)
# Create a loop to iterate through max_depth options
for i in np.arange(5, 100,5):
    dt = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=0, max_depth=7, min_samples_split=i)
    dt.fit(train_data, train_labels)
    y_preds = dt.predict(test_data)
    print ('Cohen kappa score:', metric(y_preds, test_labels), 'Max Depth: ', i)
dt = RandomForestClassifier()
#param_grid = {'criterion': ['gini', 'entropy']}
             #'max_depth': np.arange(0,15),
             #'min_samples_split': np.arange(10,100,10)}

rand_forest_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 25, 50, 85],
    'max_features': ['auto'],
    'min_samples_leaf': [10, 15, 25],
    'min_samples_split': [10, 15, 25],
    'n_estimators': [150, 200, 215]
}
dt_gridsearch = GridSearchCV(estimator=dt, param_grid = rand_forest_grid, cv = 3, n_jobs = -1,verbose = 1,scoring=scorer)
dt_gridsearch.fit(train_data, train_labels)
print(dt_gridsearch.best_params_)
metric(dt_gridsearch.predict(test_data), test_labels)
df_sub = pd.read_csv('../input/test/test.csv')
rand_forest_preds = dt_gridsearch.predict(df_sub.drop(['Name', 'Description', 'PetID', 'RescuerID'], axis=1).values)
# Store predictions for Kaggle Submission
submission_df = pd.DataFrame(data={'PetID' : df_sub['PetID'], 
                                   'AdoptionSpeed' : rand_forest_preds})
submission_df.to_csv('submission.csv', index=False)