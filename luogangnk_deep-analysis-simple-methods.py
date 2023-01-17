import pandas as pd

import numpy as np

import collections, re

import copy





from cycler import cycler

from pandas.tools.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('bmh')



%matplotlib inline  



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



from sklearn.grid_search import GridSearchCV



pd.set_option('display.max_columns', 500)

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
train.head(2)
train.describe()
train.describe(include=['O'])
## exctract cabin letter.

def extract_cabin(x):

    return x!=x and 'other' or x[0]

train['Cabin_l'] = train['Cabin'].apply(extract_cabin)
plain_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_l']

fig, ax = plt.subplots(nrows = 2, ncols = 3 ,figsize=(20,10))

start = 0

for j in range(2):

    for i in range(3):

        if start == len(plain_features):

            break

        sns.barplot(x=plain_features[start], y='Survived', data=train, ax=ax[j,i])

        start += 1
sv_lab = 'survived'

nsv_lab = 'not survived'

fig, ax = plt.subplots(figsize=(5,3))

ax = sns.distplot(train[train['Survived']==1].Age.dropna(), bins=20, label = sv_lab, ax = ax)

ax = sns.distplot(train[train['Survived']==0].Age.dropna(), bins=20, label = nsv_lab, ax = ax)

ax.legend()

_ = ax.set_ylabel('KDE')



fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

females = train[train['Sex']=='female']

males = train[train['Sex']=='male']



ax = sns.distplot(females[females['Survived']==1].Age.dropna(), bins=30, label = sv_lab, ax = axes[0], kde =False)

ax = sns.distplot(females[females['Survived']==0].Age.dropna(), bins=30, label = nsv_lab, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(males[males['Survived']==1].Age.dropna(), bins=30, label = sv_lab, ax = axes[1], kde = False)

ax = sns.distplot(males[males['Survived']==0].Age.dropna(), bins=30, label = nsv_lab, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
_ = sns.factorplot('Pclass', 'Survived', hue='Sex', col = 'Embarked', data=train)

_ = sns.factorplot('Pclass', 'Survived', col = 'Embarked', data=train)
tab = pd.crosstab(train['Embarked'],train['Pclass'])

print(tab)

tab_prop = tab.div(tab.sum(1).astype(float), axis=0)

tab_prop.plot(kind="bar", stacked=True)
ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train)

ax.set_yscale('log')
_ = sns.violinplot(x='Pclass', y='Age', hue = 'Survived', data=train, split=True)
# To get the full family size of a person, added siblings and parch.

#fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 5))

train['family_size'] = train['SibSp'] + train['Parch'] + 1 

test['family_size'] = test['SibSp'] + test['Parch'] + 1 

axes = sns.factorplot('family_size','Survived', 

                      hue = 'Sex', 

                      data=train, aspect = 4)

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(collections.Counter(train['Title']).most_common())

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print()

print(collections.Counter(test['Title']).most_common())
tab = pd.crosstab(train['Title'],train['Pclass'])

print(tab)

tab_prop = tab.div(tab.sum(1).astype(float), axis=0)

tab_prop.plot(kind="bar", stacked=True)
max(train[train['Title']== 'Master'].Age)
_ = sns.factorplot('Title','Survived', data=train, aspect = 3)
#train['Title'].replace(['Master','Major', 'Capt', 'Col', 'Countess','Dona','Lady', 'Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)

train['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)

#train['Title'].replace(['Countess','Dona','Lady'], 'titled_women', inplace = True)

#train['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled_man', inplace = True)

train['Title'].replace(['Countess','Dona','Lady'], 'Mrs', inplace = True)

#train['Title'].replace(['Master'], 'Mr', inplace = 'True')

train['Title'].replace(['Mme'], 'Mrs', inplace = True)

train['Title'].replace(['Mlle','Ms'], 'Miss', inplace = True)
g = sns.factorplot('Title','Survived', data=train, aspect = 3)
def extract_cabin(x):

    return x!=x and 'other' or x[0]

train['Cabin_l'] = train['Cabin'].apply(extract_cabin)

print(train.groupby('Cabin_l').size())

sns.factorplot('Cabin_l','Survived', 

               order = ['other', 'A','B', 'C', 'D', 'E', 'F', 'T' ], 

               aspect = 3, 

               data=train)
plt.figure(figsize=(8, 8))

corrmap = sns.heatmap(train.drop('PassengerId',axis=1).corr(),square=True, annot=True)
train.shape[0] - train.dropna().shape[0]
train.isnull().sum()
test.isnull().sum()
max_emb = np.argmax(train['Embarked'].value_counts())

train['Embarked'].fillna(max_emb, inplace=True)
indz = test['Fare'].index[test['Fare'].apply(np.isnan)].tolist

print(indz)

pclass = test['Pclass'][152]

fare_test = test[test['Pclass']==pclass].Fare.dropna()

fare_train = train[train['Pclass']==pclass].Fare

fare_med = (fare_test + fare_train).median()

print(fare_med)

test.loc[152,'Fare'] = fare_med
ages = np.concatenate((test['Age'].dropna(), train['Age'].dropna()), axis=0)

std_ages = ages.std()

mean_ages = ages.mean()

train_nas = np.isnan(train["Age"])

test_nas = np.isnan(test["Age"])

np.random.seed(122)

impute_age_train  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = train_nas.sum())

impute_age_test  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = test_nas.sum())

train["Age"][train_nas] = impute_age_train

test["Age"][test_nas] = impute_age_test

ages_imputed = np.concatenate((test["Age"],train["Age"]), axis = 0)
train['Age*Class'] = train['Age']*train['Pclass']

test['Age*Class'] = test['Age']*test['Pclass']
_ = sns.kdeplot(ages_imputed, label = 'After imputation')

_ = sns.kdeplot(ages, label = 'Before imputation')
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



test['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)

test['Title'].replace(['Countess','Dona','Lady'], 'Mrs', inplace = True)

#test['Title'].replace(['Master'], 'Mr', inplace = True)

test['Title'].replace(['Mme'], 'Mrs', inplace = True)

test['Title'].replace(['Mlle','Ms'], 'Miss', inplace = True)
train['age_cat'] = None

train.loc[(train['Age'] <= 13), 'age_cat'] = 'young'

train.loc[ (train['Age'] > 13), 'age_cat'] = 'adult'



test['age_cat'] = None

test.loc[(test['Age'] <= 13), 'age_cat'] = 'young'

test.loc[(test['Age'] > 13), 'age_cat'] = 'adult'
train_label = train['Survived']

test_pasId = test['PassengerId']

drop_cols = ['Name','Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId']

train.drop(drop_cols + ['Cabin_l'], 1, inplace = True)

test.drop(drop_cols, 1, inplace = True)
train['Pclass'] = train['Pclass'].apply(str)

test['Pclass'] = test['Pclass'].apply(str)
train.drop(['Survived'], 1, inplace = True)

train_objs_num = len(train)

dataset = pd.concat(objs=[train, test], axis=0)

dataset = pd.get_dummies(dataset)

train = copy.copy(dataset[:train_objs_num])

test = copy.copy(dataset[train_objs_num:])
droppings = ['Embarked_Q', 'Age']

#droppings += ['Sex_male', 'Sex_female']



test.drop(droppings, 1, inplace = True)

train.drop(droppings,1, inplace = True)
train.head(5)
def prediction(model, train, label, test, test_pasId):

    model.fit(train, label)

    pred = model.predict(test)

    accuracy = cross_val_score(model, train, label, cv = 5)



    sub = pd.DataFrame({

            "PassengerId": test_pasId,

            "Survived": pred

        })    

    return [model, accuracy, sub]
rf = RandomForestClassifier(n_estimators=80, min_samples_leaf = 2, min_samples_split=2, random_state=110)

acc_random_forest = prediction(rf, train, train_label, test, test_pasId)

importances = pd.DataFrame({'feature':train.columns,'importance':np.round(rf.feature_importances_,3)})

importances = importances.sort_values('importance', ascending=False).set_index('feature')

#acc_random_forest[2].to_csv('~/Desktop/random_forest.txt', index=False)

print (importances)

importances.plot.bar()

print(acc_random_forest[1])



test_predictions = acc_random_forest[0].predict(test)

test_predictions = test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": test_predictions

    })



submission.to_csv("titanic_submission_randomforest.csv", index=False)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train['Fare'].values.reshape(-1, 1))

train['Fare'] = scaler.transform(train['Fare'].values.reshape(-1, 1)) 

test['Fare'] = scaler.transform(test['Fare'].values.reshape(-1, 1))  



scaler = StandardScaler().fit(train['Age*Class'].values.reshape(-1, 1))

train['Age*Class'] = scaler.transform(train['Age*Class'].values.reshape(-1, 1)) 

test['Age*Class'] = scaler.transform(test['Age*Class'].values.reshape(-1, 1))  







lr  = LogisticRegression(random_state=110)

acc = prediction(lr, train, train_label, test, test_pasId)

print(acc[1])



test_predictions = acc[0].predict(test)

test_predictions = test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": test_predictions

    })

submission.to_csv("titanic_submission_logregres.csv", index=False)



#train.columns.tolist()

print(list(zip(acc[0].coef_[0], train.columns.tolist())))
kn = KNeighborsClassifier()

acc = prediction(kn, train, train_label, test, test_pasId)

print(acc[1])

test_predictions = acc[0].predict(test)

test_predictions = test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": test_predictions

    })

submission.to_csv("titanic_submission_kn.csv", index=False)
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[

        ('lr', lr), ('rf', rf)], voting='soft')

eclf1 = eclf1.fit(train, train_label)

test_predictions = eclf1.predict(test)

test_predictions = test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": test_predictions

    })



submission.to_csv("titanic_submission.csv", index=False)
def grid_search(clf, X, Y, parameters, cv):

    grid_model = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv)

    grid_model.fit(X, Y)

    #grid_model.cv_results_

    print("Best Score:", grid_model.best_score_," / Best parameters:", grid_model.best_params_)

    return grid_model.best_params_
param_range = np.logspace(-6, 5, 12)

parameters = dict(C= param_range, penalty = ['l1', 'l2'])

grid_search(lr, train, train_label, parameters, 5)
lr  = LogisticRegression(random_state=110, penalty= 'l1', C= 100)

acc = prediction(lr, train, train_label, test, test_pasId)

print(acc[1])



test_predictions = acc[0].predict(test)

test_predictions = test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": test_predictions

    })

submission.to_csv("titanic_submission_logregres_tuned_scaled.csv", index=False)
