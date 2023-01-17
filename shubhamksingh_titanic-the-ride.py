import numpy as np 

import pandas as pd 

import re

import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import rcParams

from scipy import stats





%matplotlib inline



rcParams['figure.figsize'] = [9, 6]

sb.set_style('white')

color = 'mediumslateblue'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

PassengerId = test['PassengerId']

sample_sub = pd.read_csv("../input/titanic/gender_submission.csv")
sample_sub.head(3)
train.head(3)
test.head(3)
train.shape, test.shape
train.info()

print("-"*50)

test.info()
list(train.columns)
display(train.isnull().sum().sort_values(ascending=False))

print("-"*50)

display(test.isnull().sum().sort_values(ascending=False))
percent_missing = (train.isnull().sum()/train.isnull().count()) * 100

percent_missing = percent_missing[percent_missing>0].sort_values(ascending=False)



display(percent_missing.to_frame(name="Percent Missing"))
sb.barplot(x=percent_missing.index, y=percent_missing, color=color)
train['Cabin'].describe()
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
train['Age'].head()
for df in [train, test]:

    age_avg = df['Age'].mean()

    age_std = df['Age'].std()

    age_null_count = df['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    df['Age'][np.isnan(df['Age'])] = age_null_random_list

    df['Age'] = df['Age'].astype(int)
test['Age'].isnull().sum()
train['Embarked'].head()
train['Embarked'].describe()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
train.isnull().sum()
test['Fare'].describe()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
display(train.isnull().sum().sort_values(ascending=False))

print("-"*50)

display(test.isnull().sum().sort_values(ascending=False))
display(train['Survived'].value_counts())
survive_data = pd.DataFrame()

survive_data['Survived'] = train['Survived'].apply(lambda x:'Died' if x==0 else 'Survived')

survive_data['Pclass'] = train['Pclass']

survive_data['Sex'] = train['Sex']



ax = sb.countplot(survive_data['Survived'], color=color)

ax.set(xlabel='Death vs Survived')
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
ax = sb.countplot(survive_data['Survived'], hue='Sex', palette='spring', data=survive_data)

ax.set(xlabel='Death vs Survived based Sex')
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
corrmat = train.corr()



sb.heatmap(corrmat, vmax=0.9, cmap="spring", annot=True, square=True, fmt='.2', linewidth='.1')
categorical_features = [col for col in train.columns

      if train[col].dtype=='object']



categorical_features
numerical_features = [col for col in train.columns

      if train[col].dtype!='object']



numerical_features
all_data = [train, test]



for df in all_data:

    df['Name_length'] = df['Name'].apply(len)

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = df['Family_size'].apply(lambda x:0 if x>1 else 1)
train.head(4)
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)
train['Fare'].head()
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)

test['Person'] = test[['Age','Sex']].apply(get_person,axis=1)
# Drop the 'Sex' variable because we don't need it anymore

train.drop(['Sex'], axis=1, inplace=True)

test.drop(['Sex'], axis=1, inplace=True)
for df in all_data:

    df['Title'] = df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
train.head(3)
train['Title'].nunique()
train['Title'].value_counts()
# Note that 'Dona' is in test data

# Ms means miss, Mlle (mademoiselle) means miss, Mme (Madame) means miss

for df in all_data:

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Lady', 'Capt', 'Jonkheer', 'Sir', 'Don', 'Dona'], 'Rare')
train['Title'].value_counts()
train.head(3)
categorical_features = [col for col in train.columns

      if train[col].dtype=='object']



categorical_features
train = train.drop(['Name', 'Ticket', 'PassengerId'], axis=1)

test = test.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
train.head()
# Store the target variable and drop it from the train dataset

X = train.drop(['Survived'], axis=1)

y = train['Survived']
# Filling numerical columns

num_cols = [col for col in X.columns if X[col].dtype!='object']

X.update(X[num_cols].fillna(0))

tst_num_cols = [col for col in test.columns if X[col].dtype!='object']

test.update(test[tst_num_cols].fillna(0))



# Filling categorical columns

cat_cols = [col for col in X.columns if X[col].dtype=='object']

X.update(X[cat_cols].fillna('None'))

test.update(test[cat_cols].fillna('None'))
# Using pd.get_dummies() to one-hot encode

X = pd.get_dummies(X)

test = pd.get_dummies(test)

X, test = X.align(test, join='left', axis=1)
X.head(3)
test.head(3)
cat = [col for col in X.columns

       if X[col].dtype=='object']

cat
tmp_all = pd.concat([X, y], axis=1)

corrmat = tmp_all.corr()



plt.subplots(figsize=(15, 15))

sb.heatmap(corrmat, vmax=0.9, square=True, annot=True, cmap='spring', fmt='.1f', linewidth='.1')
the_imp_order = corrmat['Survived'].sort_values(ascending=False).head(11).to_frame()

the_imp_order
plt.subplots(figsize=(6, 7))

plt.title('Survival Corrrelation')

sb.heatmap(the_imp_order, vmax=0.9, annot=True, fmt='.2f', cmap="spring", linewidth='.1')
numerical_data = X.select_dtypes(exclude=['object']).copy()
fig = plt.figure(figsize=(17,22))

for i in range(len(numerical_data.columns)):

    fig.add_subplot(9,4,i+1)

    sb.distplot(numerical_data.iloc[:,i].dropna(), hist=False, kde_kws={'bw':0.1}, color='mediumslateblue')

    plt.xlabel(numerical_data.columns[i])

plt.tight_layout()

plt.show()
my_facet = sb.FacetGrid(tmp_all, hue="Survived",aspect=4, palette='seismic')

my_facet.map(sb.kdeplot,'Age',shade= True)

my_facet.set(xlim=(0, tmp_all['Age'].max()))

my_facet.add_legend()
plt.subplots(figsize=(18, 7))

ax = sb.countplot(tmp_all['Age'], hue='Survived', data=tmp_all)
plt.subplots(figsize=(15, 7))

average_age = tmp_all[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()

sb.barplot(x='Age', y='Survived', data=average_age)
tmp_all[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.subplots(figsize=(11, 5))

ax = sb.countplot(survive_data['Survived'], hue='Pclass', palette='spring', data=survive_data)

ax.set(xlabel='Death vs Survived based on Ticket class')
ax = sb.pointplot('Pclass', 'Survived', color=color, data=tmp_all)
display(tmp_all[['Person_female', 'Survived']].groupby(['Person_female'], as_index=False).mean().sort_values(by='Survived', ascending=False))

display(tmp_all[['Person_male', 'Survived']].groupby(['Person_male'], as_index=False).mean().sort_values(by='Survived', ascending=False))

display(tmp_all[['Person_child', 'Survived']].groupby(['Person_child'], as_index=False).mean().sort_values(by='Survived', ascending=False))
fig, (axis1,axis2, axis3) = plt.subplots(1,3,sharex=True,figsize=(17,7))

sb.barplot(tmp_all['Person_female'], tmp_all['Survived'], order=[1,0], ax=axis1, color=color)

sb.barplot(tmp_all['Person_child'], tmp_all['Survived'], order=[1,0], ax=axis2, color=color)

sb.barplot(tmp_all['Person_male'], tmp_all['Survived'], order=[1,0], ax=axis3, color=color)
tmp_all[['Family_size', 'Survived']].groupby(['Family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
ax = sb.barplot(x=tmp_all['Family_size'], y=tmp_all['Survived'], color=color)

ax.set(xlabel='Size of the Family')
ax = sb.distplot(tmp_all['Family_size'], color=color, bins=20)

ax.set(xlabel='Size of Family', ylabel='Count')
tmp_all[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
tmp_all['IsAlone'] = tmp_all['IsAlone'].apply(lambda x: 'Alone' if x==0 else 'Not Alone')

ax = sb.barplot(x=tmp_all['IsAlone'], y=tmp_all['Survived'], color=color)

ax.set(xlabel='Is the person alone or not?')
X.head()
f = [col for col in X.columns

    if X[col].dtype!='int64']

f
change_ftr = ['Embarked_C',

 'Embarked_Q',

 'Embarked_S',

 'Person_child',

 'Person_female',

 'Person_male',

 'Title_Master',

 'Title_Miss',

 'Title_Mr',

 'Title_Mrs',

 'Title_Rare']



X[change_ftr] = X[change_ftr].astype(int)

test[change_ftr] = test[change_ftr].astype(int)
tmp_all.columns
tmp_all = pd.concat([X.copy(), y.copy()], axis=1)

sb.pairplot(tmp_all[['Survived', 'Pclass', 'Fare', 'Age', 'Name_length', 'Family_size', 'IsAlone']], hue='Survived', height=2, diag_kind='kde')
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import VotingClassifier
def rmse(model, X, y):

    scores = np.sqrt(-1 * cross_val_score(model, X, y,

                        cv=10, 

                        scoring='neg_mean_squared_error'))

    return scores



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=2)
Scores = {}
xgb = XGBClassifier(n_estimators=500,

                   learning_rate=0.01,

                   colsample_bytree=0.45,

                   max_depth=3,

                   gamma=0,

                   reg_alpha=0,

                   reg_lambda=0,

                   objective='reg:squarederror')



scores = rmse(xgb, train_X, train_y)

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))



Scores['XGB'] = scores.mean()
svc = SVC(C= 0.025, kernel='linear')





scores = rmse(svc, train_X, train_y)

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))



Scores['SVR'] = scores.mean()
gbr = GradientBoostingClassifier(n_estimators=500,

                                learning_rate=0.01,

                                max_depth=5)





scores = rmse(gbr, train_X, train_y)

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))



Scores['GBR'] = scores.mean()
lgb = LGBMClassifier(num_leaves=4,

                       learning_rate=0.01, 

                       n_estimators=900,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=3, 

                       bagging_seed=5,

                       feature_fraction=0.5,

                       feature_fraction_seed=5,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



scores = rmse(lgb, train_X, train_y)

print("Root Mean Square Error (RMSE)", str(scores.mean()))

print("Error Standard Deviation", str(scores.std()))



Scores['LGB'] = scores.mean()
# vote_set = [('svc', svc),

#             ('xgb', xgb),

#             ('lgb', lgb),

#             ('gbr', gbr),

#            ]
stack = StackingCVClassifier(classifiers=(gbr, xgb, svc, lgb),

                            meta_classifier=xgb,

                            random_state=42)
xgb.fit(train_X, train_y)
svc.fit(train_X, train_y)
gbr.fit(train_X, train_y)
lgb.fit(train_X, train_y)
stack.fit(np.array(train_X), np.array(train_y))
# voting = VotingClassifier(estimators = vote_set , voting = 'hard')

# voting.fit(train_X, train_y)
# scores = rmse(voting, train_X, train_y)

# print("Root Mean Square Error (RMSE)", str(scores.mean()))

# print("Error Standard Deviation", str(scores.std()))
def blended_predictions(X):

    return ((0.15 * svc.predict(X)) + \

            (0.15 * gbr.predict(X)) + \

            (0.15 * xgb.predict(X)) + \

            (0.05 * lgb.predict(X)) + \

           (0.4 * stack.predict(np.array(X))))
blended_score = rmsle(valid_y, blended_predictions(valid_X))



Scores['Blended'] = blended_score

blended_score
preds = np.ceil(blended_predictions(test)).astype(int)
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": preds

    })

submission.to_csv('titanic.csv', index=False)

submission.head()
submission.tail()