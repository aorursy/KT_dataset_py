import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.shape
test.shape
train.head()
#Save the 'Id' column

train_ID = train['PassengerId']

test_ID = test['PassengerId']



# Remove 'Id' for analysis

train = train.drop('PassengerId', axis=1)

test = test.drop('PassengerId', axis=1)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
train.describe()
train['Sex'].value_counts()
train['Embarked'].value_counts()
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.Survived.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['Survived'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#display top missing data ratio

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
all_data.drop(['Cabin'], axis=1, inplace=True)
#box plot Pclass/Age

var = 'SibSp'

data = pd.concat([all_data['Age'], all_data[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.boxplot(x=var, y="Age", data=data)
#box plot Pclass/Age

var = 'Pclass'

data = pd.concat([all_data['Age'], all_data[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.boxplot(x=var, y="Age", data=data)
all_data['SibSp'].loc[np.isnan(all_data['Age'])].value_counts()
all_data["Age"] = all_data.loc[all_data["SibSp"]>1].groupby("SibSp")["Age"].transform(

    lambda x: x.fillna(x.median()))
all_data["Age"] = all_data.groupby("Pclass")["Age"].transform(

    lambda x: x.fillna(x.median()))
all_data["Embarked"].value_counts()
all_data["Embarked"] = all_data["Embarked"].fillna("S")
all_data["Fare"] = all_data.groupby("Pclass")["Fare"].transform(

    lambda x: x.fillna(x.median()))
all_data["TotalRelatives"] = all_data['SibSp'] + all_data['Parch']



all_data['IsAlone'] = 1 #initialize to yes/1 is alone

all_data['IsAlone'].loc[all_data["TotalRelatives"] > 0] = 0 # now update to no/0 if family size is greater than 1



#quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

all_data['FareBin'] = pd.qcut(all_data['Fare'], 4)



#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 5)
final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
train = final_features[:ntrain]

test = final_features[ntrain:]
#Validation function

n_folds = 5



def score_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    score = cross_val_score(model, train.values, y_train, cv = kf)

    return("score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
score_cv(GaussianNB())
params = {'logisticregression__C' : [0.001,0.01,0.1,1,10,100,1000]}

pipe = make_pipeline(RobustScaler(), LogisticRegression())

gridsearch_logistic = GridSearchCV (pipe, params, cv=10)

gridsearch_logistic.fit(train, y_train)

print ("Meilleurs parametres: ", gridsearch_logistic.best_params_)
score_cv(gridsearch_logistic.best_estimator_)
params = {'kneighborsclassifier__n_neighbors' : [3,4,5,6,7],

         'kneighborsclassifier__weights' : ['uniform','distance'],

         'kneighborsclassifier__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}

pipe = make_pipeline(RobustScaler(), KNeighborsClassifier())

gridsearch_KNC = GridSearchCV (pipe, params, cv=5)

gridsearch_KNC.fit(train, y_train)

print ("Meilleurs parametres: ", gridsearch_KNC.best_params_)
score_cv(gridsearch_KNC.best_estimator_)
score_cv(XGBClassifier())
gradient = GradientBoostingClassifier()

gradient.fit(train, y_train)
score_cv(GradientBoostingClassifier())
params = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 4),

    "min_samples_leaf": np.linspace(0.1, 0.5, 4),

    "max_depth":[3,5,8],

    "max_features":["auto","log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":[100]

    }

gridsearch_gradient = RandomizedSearchCV (GradientBoostingClassifier(), params, n_iter = 500, cv=5)

gridsearch_gradient.fit(train, y_train)

print ("Meilleurs parametres: ", gridsearch_gradient.best_params_)
score_cv(gridsearch_gradient.best_estimator_)
pred = gridsearch_logistic.best_estimator_.predict(test)

sub = pd.DataFrame()

sub['PassengerID'] = test_ID

sub['Survived'] = pred

sub.to_csv('submission.csv',index=False)