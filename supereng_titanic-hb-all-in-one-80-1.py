# This Python 3 environment comes with many helpful analytics libraries installed

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set() # setting seaborn default for plots

%matplotlib inline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as mc

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

import sklearn.ensemble as ens 

# !pip install xgboost

import xgboost as xgb 

#!pip install lightgbm

import lightgbm as lgbm

#!pip install catboost

import catboost as cb

from mlxtend.classifier import StackingClassifier

from tpot import TPOTClassifier
TRAIN_PATH = '../input/train.csv'

TEST_PATH = '../input/test.csv'

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Survived'].value_counts()
def bar_chart(feature):

    survived = train_df[train_df['Survived']==1][feature].value_counts()

    dead = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
bar_chart('Pclass')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

train_df.columns
train_test_data = [train_df, test_df] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    
train_df.columns
test_df.columns
# train_df['Has_family'] = train_df['SibSp'] + train_df['Parch']

# test_df['Has_family'] = test_df['SibSp'] + test_df['Parch']

# train_df[train_df['Has_family']>0] = 1

# test_df[test_df['Has_family']>0] = 1
X_train_df = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])

X_test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
X_train_df.shape, X_test_df.shape
y_train_df = train_df['Survived']
X_train_df.isnull().sum()
X_test_df.isnull().sum()
# X_train_df.loc[ X_train_df['Age'] <= 5, 'Age']= 0

# X_train_df.loc[(X_train_df['Age'] > 5) & (X_train_df['Age'] <= 16), 'Age'] = 1

# X_train_df.loc[(X_train_df['Age'] > 16) & (X_train_df['Age'] <= 32), 'Age'] = 2

# X_train_df.loc[(X_train_df['Age'] > 32) & (X_train_df['Age'] <= 48), 'Age'] = 3

# X_train_df.loc[(X_train_df['Age'] > 48) & (X_train_df['Age'] <= 64), 'Age'] = 4

# X_train_df.loc[ X_train_df['Age'] > 64, 'Age'] = 5

# X_train_df
# X_test_df.loc[ X_test_df['Age'] <= 5, 'Age']= 0

# X_test_df.loc[(X_test_df['Age'] > 5) & (X_test_df['Age'] <= 16), 'Age'] = 1

# X_test_df.loc[(X_test_df['Age'] > 16) & (X_test_df['Age'] <= 32), 'Age'] = 2

# X_test_df.loc[(X_test_df['Age'] > 32) & (X_test_df['Age'] <= 48), 'Age'] = 3

# X_test_df.loc[(X_test_df['Age'] > 48) & (X_test_df['Age'] <= 64), 'Age'] = 4

# X_test_df.loc[ X_test_df['Age'] > 64, 'Age'] = 5
mod = X_train_df.Embarked.value_counts().argmax()

X_train_df.Embarked.fillna(mod, inplace=True)
fare_med = train_df.Fare.median()

X_test_df.Fare.fillna(fare_med, inplace=True)
age_med = train_df.Age.median()

X_train_df.Age.fillna(age_med, inplace=True)

X_test_df.Age.fillna(age_med, inplace=True)
X_train_df.isnull().sum()
X_test_df.isnull().sum()
X_train_df = pd.get_dummies(X_train_df, columns=['Sex', 'Pclass', 'Embarked', 'Title'], drop_first=True)

X_test_df = pd.get_dummies(X_test_df, columns=['Sex', 'Pclass', 'Embarked', 'Title'], drop_first=True)
X_train_df.head()
X_test_df.head()
X_train_df.shape, X_test_df.shape
# age_notnull = pd.concat([X_train_df[X_train_df.Age.notnull()], X_test_df[test_df.Age.notnull()]], axis=0)

# X_age_notnull = age_notnull.drop(columns=['Age'])

# y_age_notnull = pd.DataFrame(age_notnull['Age'])

# X_age_train_null = X_train_df[X_train_df.Age.isnull()].drop(columns=['Age'])

# X_age_test_null = X_test_df[X_test_df.Age.isnull()].drop(columns=['Age'])

# #------------------------------------------------------------------------

# #  Data Scalling

# age_sc_X = MinMaxScaler()

# X_age_notnull = age_sc_X.fit_transform(X_age_notnull)

# X_age_train_null = age_sc_X.transform(X_age_train_null)

# X_age_test_null = age_sc_X.transform(X_age_test_null)

# age_sc_y = MinMaxScaler()

# y_age_notnull = age_sc_y.fit_transform(y_age_notnull)

# #------------------------------------------------------------------------

# #  Age Predecttion

# X_train, X_test, y_train, y_test = train_test_split(X_age_notnull, y_age_notnull.ravel(), test_size=0.2, random_state=42)

# RF_age_reg = ens.RandomForestRegressor(n_estimators= 845, min_samples_split= 5,min_samples_leaf= 2,

#                                        max_features= 'sqrt', max_depth= 100, bootstrap= True)

# RF_age_reg.fit(X_train, y_train)

# y_pred = RF_age_reg.predict(X_test)

# mc.mean_absolute_error(y_test, y_pred), mc.r2_score(y_test, y_pred)

# X_age_train_null_pred = list(RF_age_reg.predict(X_age_train_null))

# X_age_test_null_pred = list(RF_age_reg.predict(X_age_test_null))

# train_null_index = list(X_train_df.Age[X_train_df.Age.isnull()].index)

# test_null_index = list(X_test_df.Age[X_test_df.Age.isnull()].index)

# train_fillna_df = pd.DataFrame(X_age_train_null_pred, columns=['Age'], index=train_null_index)

# test_fillna_df = pd.DataFrame(X_age_test_null_pred, columns=['Age'], index=test_null_index)

# X_train_df.Age.fillna(train_fillna_df['Age'], inplace=True)

# X_test_df.Age.fillna(test_fillna_df['Age'], inplace=True)

# X_train_sc = X_train_df

# X_test_sc = X_test_df
sc_X = MinMaxScaler()

X_train_sc = sc_X.fit_transform(X_train_df)

X_test_sc = sc_X.transform(X_test_df)
X_train, X_test, y_train, y_test = train_test_split(X_train_sc, y_train_df.values, test_size=0.2)
logi_clf = LogisticRegression(solver='lbfgs', max_iter=1000)

logi_parm = {"C": [0.1, 0.5, 1, 5],

             'solver': ['newton-cg', 'lbfgs'],

}
svm_clf = SVC(probability=True)

svm_parm = {'kernel': ['rbf', 'poly'], 

            'C': [1, 100, 500], 

            'gamma': ['scale']}
dt_clf = DecisionTreeClassifier()

dt_parm = {'criterion':['gini', 'entropy']}
knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 

            'weights':['uniform', 'distance'], 

            'p': [1,2]}
gnb_clf = GaussianNB()

gnb_parm = {'var_smoothing':[1e-09]}
def generate_submission(clf, file_name):

    pred = clf.predict(X_test_sc)

    pred = np.array(pred, dtype='int')

    test_df = pd.read_csv(TEST_PATH)

    y_test_df = test_df['PassengerId']

    cols = ['PassengerId', 'Survived']

    submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 

                             columns=cols)

    submit_df.to_csv('submission_{}.csv'.format(file_name), index=False)
clfs = [logi_clf, svm_clf, dt_clf, knn_clf, gnb_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm, gnb_parm] 

clf_names = ['logistic', 'SVM', 'DT', 'KNN', 'GNB']
clfs_opt = []

clfs_best_scores = []

clfs_best_param = []

i=0

for clf_, param in zip(clfs, params):

    clf = RandomizedSearchCV(clf_, param, cv=5)

    clf.fit(X_train, y_train)

    clfs_opt.append(clf.best_estimator_)

    clfs_best_scores.append(clf.best_score_)

    clfs_best_param.append(clf.best_params_)

    print(i)

    i+=1
all_Clfs_dict = {}

all_Clfs_list = []

for name, clf in zip(clf_names, clfs_opt):

    all_Clfs_dict[name] = clf

    all_Clfs_list.append((name, clf))
arg = np.argmax(clfs_best_scores)
clf = clfs_opt[arg]

y_pred = clf.predict(X_test)

mc.accuracy_score(y_test, y_pred)
voting_clf = ens.VotingClassifier(all_Clfs_list)

voting_param = {'voting': ['hard', 'soft'], 'weights': [clfs_best_scores]}
meta_est_clf = ens.BaggingClassifier(base_estimator=clfs_opt[arg])

meta_est_param = {'n_estimators':range(50,1000,100),

        'max_samples':[1.0, 0.9, 0.8],

        'bootstrap_features':[False, True],

        'random_state': [0,1,2,3,4,5]}
rf_clf = ens.RandomForestClassifier()

rf_param = {'n_estimators' : range(50,1000,100),

         'criterion': ['gini', 'entropy'],

         'max_depth' : range(4,16,2),

         'min_samples_split' : [2, 5, 10],

         'min_samples_leaf' : [1, 2, 4],

         'bootstrap' : [True, False]

        }
ada_clf = ens.AdaBoostClassifier()

ada_param = {'n_estimators' : range(50,1000,100),

            'learning_rate': [1.0, 0.01, 0.001]

        }
gbm_clf = ens.GradientBoostingClassifier()

gbm_param = {'n_estimators' : range(50,1000,100),

            'max_depth':range(4,16,2), 

        }
xgbm_param = {'n_estimators' : range(50,1000,100),

            'learning_rate': [1.0, 0.01, 0.001],

              'objective': ['binary:logistic'],

        }

xgbm_clf = xgb.XGBClassifier()
lgbm_param = {'n_estimators' : range(50,1000,100),

            'learning_rate': [1.0, 0.01, 0.001],

        }

lgbm_clf = lgbm.LGBMClassifier()
ens_clfs = [rf_clf, ada_clf, gbm_clf, xgbm_clf, lgbm_clf]

ens_params = [rf_param, ada_param, gbm_param, xgbm_param, lgbm_param] 

ens_clf_names = ['RF', 'Ada', 'GBM', 'XGBM', 'LGBM']
ens_clfs_opt = []

ens_clfs_best_scores = []



i=0

for clf_, param in zip(ens_clfs, ens_params):

    clf = RandomizedSearchCV(clf_, param, cv=5)

    clf.fit(X_train, y_train)

    ens_clfs_opt.append(clf.best_estimator_)

    ens_clfs_best_scores.append(clf.best_score_)

    print(i)

    i+=1
arg = np.argmax(ens_clfs_best_scores)

ens_clfs_best_scores[arg]

# clf = clfs_opt[arg]
ens_clf_names[arg]
clf = ens_clfs_opt[arg]

y_pred = clf.predict(X_test)

mc.accuracy_score(y_test, y_pred)
generate_submission(clf, 'best_ens')
cat_clf = cb.CatBoostClassifier(

            iterations = 1000,

            learning_rate= 0.01,

)
cat_clf.fit(X_train, y_train, early_stopping_rounds=5, use_best_model=True, plot=True,

           eval_set=(X_test, y_test), verbose_eval=False)
cat_clf.get_best_score()
y_pred = cat_clf.predict(X_test)

mc.accuracy_score(y_test, y_pred)
generate_submission(cat_clf, 'CatBoost')
# ens_clfs_opt.append(cat_clf)

# ens_clfs_best_scores.append(cat_clf.get_best_score())
stackin_clf = StackingClassifier(classifiers=ens_clfs_opt, meta_classifier=cat_clf)

stackin_clf.fit(X_train_sc, y_train_df)
pred=stackin_clf.predict(X_test)

mc.accuracy_score(y_test, y_pred)
generate_submission(stackin_clf, 'stacking')
tpot_clf = TPOTClassifier(generations=50,population_size=100, 

                          crossover_rate=0.5, mutation_rate=0.1,

                          verbosity=False)
tpot_clf.fit(X_train, y_train)

tpot_clf.score
pred=tpot_clf.predict(X_test)

mc.accuracy_score(y_test, y_pred)
generate_submission(tpot_clf, 'tpot')
tpot_clf.export('tpot_titanic_pipline.py')
# %load tpot_titanic_pipline.py