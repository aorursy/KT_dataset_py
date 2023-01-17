import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import model_selection

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV



from sklearn import model_selection, metrics

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier ,RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from mlxtend.classifier import StackingCVClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from xgboost import XGBClassifier





import warnings

warnings.filterwarnings("ignore")



seed = 42

kfold = StratifiedKFold(n_splits=10)
df_train = pd.read_csv("../input/titanic/train.csv", index_col = 0)

df_test = pd.read_csv("../input/titanic/test.csv", index_col = 0)



df_all = pd.concat([df_train, df_test])
df_all.dtypes.sort_values()
print('Number of observations:', len(df_all), '\n')

print('Unique values:')

print(df_all.nunique().sort_values(ascending = False))
df_all.describe()
def missing(df):

    df_missing = pd.DataFrame(df.isna().sum().sort_values(ascending = False), columns = ['missing_count'])

    df_missing['missing_share'] = df_missing.missing_count / len(df_train)

    return df_missing
missing(df_train)
missing(df_test)
def simple_chart(df, x, title, hue = None):

    plt.figure(figsize = (10, 6))

    plt.title(title, fontsize=14)

    ax = sns.countplot(x = x, hue = hue, data = df)
def multi_variable_chart(df, vhue, vars_, title = None):

    fig, axs = plt.subplots(ncols=len(vars_), figsize=(15, 5))

    for i in range(len(vars_)):

        sns.countplot(x = vars_[i], hue = vhue, data = df, ax=axs[i])

    fig.suptitle(title, fontsize=14)
def dist_chart(df, x, hue, rows=None):

    g = sns.FacetGrid(df, hue = hue, row = rows, aspect = 4)

    g.map(sns.kdeplot, x, shade = True)

    g.set(xlim=(0, df[x].max()))

    g.add_legend()
sns.set(style="darkgrid")
simple_chart(df_all, x = 'Survived', title = 'Passenger Survival Count')
simple_chart(df_all, x = 'Sex', hue = 'Survived', title = 'Survival by Sex')
multi_variable_chart(df_all, vhue= 'Sex', vars_ = ('Pclass', 'Embarked', 'Parch', 'SibSp'), 

                     title = 'Sex distribution for all data')
multi_variable_chart(df_all[df_all.Sex == 'female'], vhue= 'Survived', 

               vars_ = ('Pclass', 'Embarked', 'Parch', 'SibSp'), title = 'Survival of females by features')

multi_variable_chart(df_all[df_all.Sex == 'male'], vhue= 'Survived', 

               vars_ = ('Pclass', 'Embarked', 'Parch', 'SibSp'), title = 'Survival of males by features')
simple_chart(df_all, x = 'Pclass', hue = 'Survived', title = 'Survival by Pclass')
multi_variable_chart(df_all, vhue= 'Pclass', 

               vars_ = ('Sex', 'Embarked', 'Parch', 'SibSp'), title = 'Pclass distribution by features')
multi_variable_chart(df_train[df_train.Pclass == 1], vhue= 'Survived', 

               vars_ = ('Sex', 'Embarked', 'Parch', 'SibSp'), title = 'Sirvival of Pclass = 1 by features')

multi_variable_chart(df_train[df_train.Pclass == 2], vhue= 'Survived', 

               vars_ = ('Sex', 'Embarked', 'Parch', 'SibSp'), title = 'Sirvival of Pclass = 2 by features')

multi_variable_chart(df_train[df_train.Pclass == 3], vhue= 'Survived', 

               vars_ = ('Sex', 'Embarked', 'Parch', 'SibSp'), title = 'Sirvival of Pclass = 3 by features')
df_all[['Name']].head(7)
df_all['Title'] = df_all['Name'].str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0].strip())
df_all['Title'].value_counts()
simple_chart(df_all[df_all.Title.isin(['Mr', 'Mrs', 'Miss', 'Master'])], 

                    x = 'Title', hue = 'Survived', title = 'Survival by major Title')

simple_chart(df_all[~df_all.Title.isin(['Mr', 'Mrs', 'Miss', 'Master'])], 

                    x = 'Title', hue = 'Survived', title = 'Survival by minor Title')
plt.figure(figsize = (10, 6))

ax = sns.kdeplot(df_all.Fare)
ax = sns.factorplot(x = 'Survived', y = 'Fare', data = df_all[df_all.Fare < 200], kind = 'box', size=6)
ax = sns.factorplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = df_all[df_all.Fare < 200], 

                    kind = 'box', size= 6)
dist_chart(df = df_all[df_all.Fare < 100], x = 'Fare', hue = 'Survived', rows = 'Pclass')
ax = sns.factorplot(x = 'Sex', y = 'Fare', data = df_all, 

                    kind = 'box', size= 5, aspect = 1)
ax = sns.factorplot(x = 'SibSp', y = 'Fare', data = df_all, 

                    kind = 'box', size= 6, aspect = 2)
ax = sns.factorplot(x = 'Parch', y = 'Fare', data = df_all, 

                    kind = 'box', size= 6, aspect = 2)
plt.figure(figsize = (10, 6))

ax = sns.distplot(df_all.Age, bins = 20) 
ax = sns.factorplot(x = 'Survived', y = 'Age', data = df_all, kind = 'box', size=6)
dist_chart(df_all, x = 'Age', hue = 'Survived', rows = 'Sex')
dist_chart(df_all, x = 'Age', hue = 'Survived', rows = 'Pclass')
ax = sns.factorplot(x = 'Title', y = 'Age', data = df_all, kind = 'box', size=6, aspect = 2)
df_all['Cabin_Liter'] = df_all['Cabin'].str[0]

df_train['Cabin_Liter'] = df_train['Cabin'].str[0]

df_test['Cabin_Liter'] = df_test['Cabin'].str[0]
plt.figure(figsize = (10, 6))

ax = sns.countplot(x = 'Cabin_Liter', data = df_all)



total = len(df_all['Cabin_Liter'])



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/total), (x.mean(), y), 

    ha='center', va='bottom')



plt.show()
plt.figure(figsize = (10, 6))

ax = sns.countplot(x = 'Cabin_Liter', hue = 'Survived', data = df_train)



total = len(df_train['Cabin_Liter'])



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/total), (x.mean(), y), 

    ha='center', va='bottom')



plt.show()
simple_chart(df_all, x = 'Cabin_Liter', hue = 'Pclass', title = 'Cabin Liter by Pclass')
df_all['Cabin_Known'] = 1

df_all.loc[df_all.Cabin_Liter.isna() == True, 'Cabin_Known'] = 0
simple_chart(df_all, x = 'Cabin_Known', hue = 'Survived', title = 'Survival by Cabin_Known')
df_all[df_all.Age.isna() == True].Title.value_counts()
df_all.groupby('Title').Age.median()
df_all.groupby(['Title', 'Parch']).Age.median()
df_all.loc[((df_all['Title'] == 'Miss') & (df_all['Parch'] > 0)),'Title'] = "Young_Miss"
df_all["Age"] = df_all.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
df_all['Title'] = df_all["Title"].replace(['Lady', 'Mme', 'Dona', 'the Countess'], 'Mrs')

df_all['Title'] = df_all["Title"].replace(['Mlle', 'Ms'], 'Miss')

df_all['Title'] = df_all["Title"].replace(['Col', 'Capt', 'Don', 'Jonkheer', 'Major', 'Rev', 'Dr', 'Sir'], 'Special_Titles')
df_all['Title_Master'] = 0

df_all.loc[df_all.Title == 'Master', 'Title_Master'] = 1

df_all['Title_Miss'] = 0

df_all.loc[df_all.Title == 'Miss', 'Title_Miss'] = 1

df_all['Title_Mr'] = 0

df_all.loc[df_all.Title == 'Mr', 'Title_Mr'] = 1

df_all['Title_Mrs'] = 0

df_all.loc[df_all.Title == 'Mrs', 'Title_Mrs'] = 1

df_all['Title_Young_Miss'] = 0

df_all.loc[df_all.Title == 'Young_Miss', 'Title_Young_Miss'] = 1

df_all['Title_Special'] = 0

df_all.loc[df_all.Title == 'Special_Titles', 'Title_Special'] = 1
df_all['Family_Size'] = df_all['Parch'] + df_all['SibSp'] 
simple_chart(df_all, x = 'Family_Size', hue = 'Survived', title = 'Survival by Family_Size')
df_all['Ticket_group_size'] = 1

for num, gr in df_all.groupby('Ticket'):

    df_all.loc[df_all['Ticket'] == num, 'Ticket_group_size'] = len(gr)
simple_chart(df_all, x = 'Ticket_group_size', hue = 'Survived', title = 'Survival by Ticket_group_size')
df_all['Lone_Ticket'] = 0

df_all.loc[df_all.Ticket_group_size == 1, 'Lone_Ticket'] = 1

df_all['Nuclear_Ticket'] = 0

df_all.loc[df_all.Ticket_group_size == 2, 'Nuclear_Ticket'] = 1

df_all['Small_group_Ticket'] = 0

df_all.loc[df_all.Ticket_group_size.isin([3, 4]), 'Small_group_Ticket'] = 1

df_all['Large_group_Ticket'] = 0

df_all.loc[df_all.Ticket_group_size > 4, 'Large_group_Ticket'] = 1
fig, axs = plt.subplots(figsize=(10, 6))

sns.violinplot(x = 'Ticket_group_size', y = 'Fare', data = df_all[df_all.Pclass == 1])

fig, axs = plt.subplots(figsize=(10, 6))

sns.violinplot(x = 'Ticket_group_size', y = 'Fare', data = df_all[df_all.Pclass == 2])

fig, axs = plt.subplots(figsize=(10, 6))

sns.violinplot(x = 'Ticket_group_size', y = 'Fare', data = df_all[df_all.Pclass == 3])
gr_count = 0

for num, gr in df_all.groupby('Ticket'):

    if gr_count < 3:

        if gr.Ticket_group_size.mean() > 1:

            print(gr.Fare)

            gr_count  = gr_count + 1
df_all['Fare_Personal'] = df_all['Fare'] / df_all['Ticket_group_size']
fig, axs = plt.subplots(figsize=(10, 6))

sns.violinplot(x = 'Ticket_group_size', y = 'Fare_Personal', data = df_all)
plt.figure(figsize = (10, 6))

ax = sns.distplot(df_all.Fare_Personal, bins = 20)
dist_chart(df_all[df_all.Fare_Personal < 60], x = 'Fare_Personal', hue = 'Survived')
df_all[df_all.Fare.isna()]
df_all.groupby(['Pclass', 'Embarked']).Fare_Personal.median()
df_all.loc[1044, 'Fare'] = 7.7958

df_all.loc[1044, 'Fare_Personal'] = 7.7958
df_all['Fare_Bins'] = pd.qcut(df_all['Fare'], 20, labels=list(range(0, 20))).astype('int32')
simple_chart(df_all, x = 'Fare_Bins', hue = 'Survived', title = 'Survival by Fare_Bins')

simple_chart(df_all, x = 'Fare_Bins', hue = 'Pclass', title = 'Pclass by Fare_Bins') 
df_all['10_binned_Fare'] = 0

df_all.loc[(df_all['Fare_Bins'] >=2) & (df_all['Fare_Bins'] <= 3), '10_binned_Fare'] = 1

df_all.loc[(df_all['Fare_Bins'] >= 4) & (df_all['Fare_Bins'] <= 7), '10_binned_Fare'] = 2

df_all.loc[(df_all['Fare_Bins'] == 8), '10_binned_Fare'] = 3

df_all.loc[(df_all['Fare_Bins'] >= 9) & (df_all['Fare_Bins'] <= 10), '10_binned_Fare'] = 4

df_all.loc[(df_all['Fare_Bins'] >= 11) & (df_all['Fare_Bins'] <= 12), '10_binned_Fare'] = 5

df_all.loc[(df_all['Fare_Bins'] == 13), '10_binned_Fare'] = 6

df_all.loc[(df_all['Fare_Bins'] == 14), '10_binned_Fare'] = 7

df_all.loc[(df_all['Fare_Bins'] == 15), '10_binned_Fare'] = 8

df_all.loc[(df_all['Fare_Bins'] >= 16) & (df_all['Fare_Bins'] <= 17), '10_binned_Fare'] = 9

df_all.loc[(df_all['Fare_Bins'] >= 18), '10_binned_Fare'] = 10
simple_chart(df_all, x = '10_binned_Fare', hue = 'Survived', title = 'Survival by 10_binned_Fare') 
df_all['Age_Bins'] = 0

df_all.loc[(df_all['Age'] > 6) & (df_all['Age'] <= 14), 'Age_Bins'] = 1

df_all.loc[(df_all['Age'] > 14) & (df_all['Age'] <= 30), 'Age_Bins'] = 2

df_all.loc[(df_all['Age'] > 30) & (df_all['Age'] <= 55), 'Age_Bins'] = 3

df_all.loc[df_all['Age'] > 55, 'Age_Bins'] = 4

df_all['Age_Bins'] = df_all['Age_Bins'].astype(int)
simple_chart(df_all, x = 'Age_Bins', hue = 'Survived', title = 'Survival by Age_Bins')
df_all['Ticket_Survival'] = "Unknown"

for num, gr in df_all.groupby('Ticket'):

    jsurv, jdied, jnan = 0, 0, 0

    if len(gr) > 2:

        for i, j in gr.iterrows():

            if j.Survived == 1:

                jsurv += 1 # count survivors in the group

            elif j.Survived == 0:

                jdied += 1 # count perished in the group

            else:

                jnan +=1 # count unknown suvival in the group

    if (jnan < 2) & (jsurv+jdied != 0):

        if jsurv/(jsurv+jdied) >= 0.5:

            df_all.loc[df_all['Ticket'] == num, 'Ticket_Survival'] = "LikelyLive"

        else:

            df_all.loc[df_all['Ticket'] == num, 'Ticket_Survival'] = "LikelyDie"
df_all['Ticket_LikelyLive'] = 0

df_all.loc[df_all.Ticket_Survival == 'LikelyLive', 'Ticket_LikelyLive'] = 1

df_all['Ticket_LikelyDie'] = 0

df_all.loc[df_all.Ticket_Survival == 'LikelyDie', 'Ticket_LikelyDie'] = 1
simple_chart(df_all, x = 'Ticket_Survival', hue = 'Survived', title = 'Survival by Ticket_Survival') 
y = df_all[~df_all.Survived.isna()]['Survived'].copy()

train_0 = df_all[~df_all.Survived.isna()].drop('Survived', axis = 1).copy()

test_0 = df_all[df_all.Survived.isna()].drop('Survived', axis = 1).copy()
train_1 = train_0[['Pclass', 'Age_Bins', 'Small_group_Ticket', 'Ticket_LikelyLive', 'Ticket_LikelyDie',

                   'Family_Size', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Fare',

                   'Title_Young_Miss', 'Cabin_Known', '10_binned_Fare', 'Lone_Ticket', 'Large_group_Ticket']].copy()

test_1 = test_0[['Pclass', 'Age_Bins', 'Small_group_Ticket', 'Ticket_LikelyLive', 'Ticket_LikelyDie',

                   'Family_Size', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Fare',

                   'Title_Young_Miss', 'Cabin_Known', '10_binned_Fare', 'Lone_Ticket', 'Large_group_Ticket']].copy()
#AdaBoost

DTC = DecisionTreeClassifier(random_state=seed)

adaDTC = AdaBoostClassifier(DTC, random_state=seed)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(train_1,y)

ada_best = gsadaDTC.best_estimator_

print("AdaBoost Best Score: ", gsadaDTC.best_score_)





#ExtraTrees

ExtC = ExtraTreesClassifier(random_state=seed)

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(train_1, y)

ExtC_best = gsExtC.best_estimator_

print("ExtraTrees Best Score: ", gsExtC.best_score_)



#RandomForest

RFC = RandomForestClassifier(random_state=seed)

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(train_1, y)

RFC_best = gsRFC.best_estimator_

print("RandomForest Best Score: ", gsRFC.best_score_)



#GradientBoosting

GBC = GradientBoostingClassifier(random_state=seed)

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1]}

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_1, y)

GBC_best = gsGBC.best_estimator_

print("GradientBoosting Best Score: ", gsGBC.best_score_)



#SVC

SVMC = SVC(probability=True, random_state=seed)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(train_1, y)

SVMC_best = gsSVMC.best_estimator_

print("SVC Best Score: ", gsSVMC.best_score_)
LogRC = LogisticRegressionCV(cv=10, random_state=seed)

LogRC.fit(train_1, y)



LDA = LinearDiscriminantAnalysis()

LDA.fit(train_1, y)
grid_values = {#'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

        #'gamma': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 1, 2, 5, 7, 10],

        #'n_estimators': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000], 

        #'learning_rate': [0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3],

        #'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],

        #'colsample_bytree': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        }

params_fixed = {

    'learning_rate': 0.01,

    'gamma': 0,

    'max_depth': 5,

    'n_estimators': 1400,

    'subsample': 0.6,

    'colsample_bytree': 0.8,

    'objective': 'binary:logistic',

    'silent': True,

    'random_state': seed

    }

grid_xgb = GridSearchCV(XGBClassifier(**params_fixed, early_stopping_rounds=15), 

                         param_grid = grid_values, scoring = 'accuracy')



grid_xgb.fit(train_1, y)



xgb_best = grid_xgb.best_estimator_

print(grid_xgb.best_score_)

print(xgb_best)
xgb_best = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.05,

              early_stopping_rounds=15, gamma=1,

              learning_rate=0.2, max_delta_step=0, max_depth=7,

              min_child_weight=1, missing=None, n_estimators=4000, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=seed,

              silent=True, subsample=0.6, verbosity=1)
xgb_best.fit(train_1,y)
sclf = StackingCVClassifier(classifiers=[ada_best, ExtC_best, RFC_best, GBC_best, SVMC_best, LogRC, LDA],

                            use_probas=False,

                            meta_classifier=xgb_best,

                            random_state=seed)
sclf.fit(train_1, y)

y_preds = sclf.predict(test_1)



y_stack = pd.DataFrame()

y_stack['PassengerId'] = test_1.index

y_stack['Survived'] = y_preds.astype('int32')

y_stack[:15]
# y_stack.to_csv('submit.csv', index=False)