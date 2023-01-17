# import needed modules

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#import graphviz



# ignore seaborn warnings

import warnings

warnings.filterwarnings("ignore")





from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



import xgboost as xgb
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# inspect the first 5 rows

train.head()
print(train.info())

print(train.describe())
# see whether fare has outliers

sns.set()

sns.distplot(train[('Fare')]);
sns.violinplot(x="Survived", y="Fare", data=train);
# Gender and survival data, numerical presentations showing counts and percentages

print(pd.crosstab(train['Survived'], train['Sex'], margins=True))

print('\n')

print(pd.crosstab(train['Survived'], train['Sex'], margins=True, normalize=True))
# Gender and survival graphic presentation; much easier to understand

sns.barplot(x=('Sex'), y=('Survived'), data=train);
sns.catplot(x=("Pclass"), y=("Survived"), kind="bar", data=train);
sns.catplot(x=("Sex"), y=("Survived"), hue="Pclass", kind="bar", data=train);
# age distribution of passengers

sns.set()

ax = sns.distplot(train[('Age')].dropna())

ax.set_title('Age of Passengers');
# Ages of Non-Survivors and Survivors

ax = sns.FacetGrid(train, col ='Survived', height=8, aspect=1);

ax.map(sns.distplot, ("Age"))

fig = ax.fig

fig.suptitle('Age of Non-Survivors and Survivors', fontsize=16)

plt.subplots_adjust(top = 0.9);
sns.catplot(x=("Embarked"), y=("Survived"), kind="bar", data=train);
# look at siblings/spouses

sns.catplot(x=("SibSp"), y=("Survived"), kind="bar", data=train);

# look at parents/children

sns.catplot(x=("Parch"), y=("Survived"), kind="bar", data=train);
# save train and test lens

train_len = len(train)

test_len = len(test)



# save train survived

train_survived = train['Survived']



# drop train survived

train.drop(columns=['Survived'], inplace=True)





print(train.shape, test.shape)

print(train_survived.shape)
# combine train and test

dataset = pd.concat([train, test], axis=0, ignore_index=True)



print(dataset.shape)

dataset.tail()
dataset.drop(['Name', 'Ticket', 'Cabin', 'Fare'], axis=1, inplace=True)

dataset.head()
print(dataset.isnull().sum())

dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

# note that fillna with "mode" has a different signature than mean and median because there could be multiple modes

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

print('\n', dataset.isnull().sum())
dataset.head()
dataset['Age_Binned'] = pd.cut(dataset['Age'], [0, 5, 80], labels=['0-5', '5-80'])

dataset.drop(['Age'], inplace=True, axis=1)

dataset.head()
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked', 'Age_Binned'])

dataset.head()
train_survived.head()
train_processed = dataset[:train_len]

train_processed = pd.concat([train_processed, train_survived], axis=1)



test_processed = dataset[train_len:]



print(train_processed.shape)

print(test_processed.shape)
# Create target object

y = train_processed['Survived']



# Create the predictor variables; strip off PassengerId and Survived

X = train_processed.iloc[:,1:-1]



# Split into training and testing data

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
# Define and fit the model, set n_neighbors at 5

knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(train_X, train_y)



print(knn_model.score(X=test_X, y=test_y))
# Train and Test the model using cross-validation with 5 folds

cv_scores = cross_val_score(knn_model, train_X, train_y, cv=5)



print(cv_scores)

print('cv_scores_mean: {}'.format(np.mean(cv_scores)))

# Hypertune parameters using GridSearchCV



# create new knn model

knn2 = KNeighborsClassifier()



# create a dictionary of n_neighbor values to assess

param_grid = {'n_neighbors': np.arange(1,25)}



# use gridsearch to test all values for n_neighbors

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)



# fit model to data

knn_gscv.fit(train_X, train_y)



# check top performing n_neighbors value

print(knn_gscv.best_params_)



# check mean score for the top performing value of n_neighbors

print(knn_gscv.best_score_)
# Define and fit the model

dtc_model = DecisionTreeClassifier(random_state=1, max_depth=3)

dtc_model.fit(train_X,train_y)





feature_importance = dict(zip(test_X.columns, dtc_model.feature_importances_))

print('Feature Importance: ', feature_importance)

print('\n')

print(max(feature_importance.values()))

print('\n')

print(dtc_model.score(X=test_X, y=test_y))
# Hypertune the parameters using GridSearchCV





tree_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

dtc2 = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=5)

dtc2.fit(train_X, train_y)



# check best parameters

print(dtc2.best_params_)



# check mean score for the top performing parameters

print(dtc2.best_score_)
# Define and fit the model

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(train_X, train_y)



print(rf_model.score(X=train_X, y=train_y))
# Train and Test the model using cross-validation with 5 folds

cv_scores = cross_val_score(rf_model, train_X, train_y, cv=5)



print(cv_scores)

print('cv_scores_mean: {}'.format(np.mean(cv_scores)))
# Define and fit the model

lr_model = LogisticRegression(solver='lbfgs', random_state=1)

lr_model.fit(train_X,train_y)



print(lr_model.score(X=test_X, y=test_y))
# Train and Test the model using cross-validation with 5 folds

cv_scores = cross_val_score(lr_model, train_X, train_y, cv=5)



print(cv_scores)

print('cv_scores_mean: {}'.format(np.mean(cv_scores)))
# Define and fit the model

xgb_model = xgb.XGBClassifier()

xgb_model.fit(train_X, train_y)



print(xgb_model.score(X=test_X, y=test_y))
# apply the best hyperparameters, fit the model, and get the score:



rf = RandomForestClassifier(criterion='entropy', 

                             n_estimators=1000,

                             min_samples_split=12,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(train_X, train_y)

print(rf.oob_score_)
pd.concat((pd.DataFrame(train_processed.iloc[:, 1:-1].columns, columns = ['Feature']), 

           pd.DataFrame(rf.feature_importances_, columns = ['Importance'])), 

          axis = 1).sort_values(by='Importance', ascending = False)[:20]
# make predictions which we will submit.  Use the tuned DecisionTreeClassifier model



test_preds = dtc2.predict(test_processed.iloc[:,1:])



output = pd.DataFrame({'PassengerId': test_processed['PassengerId'],

                       'Survived': test_preds})



output.to_csv('titanic_submission_dtc2_2019_02_16.csv', index=False)