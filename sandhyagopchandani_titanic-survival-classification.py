# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import Imputer



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from subprocess import check_output

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
percent_missing = train_df.isnull().sum() * 100 / len(train_df)

missing_value_df = pd.DataFrame({'column_name': train_df.columns,

                                 'percent_missing': percent_missing})

missing_value_df
# Cabin variable is 77% missing. 

# Age is about 20% missing

# Port of Embarkation is missing for two pessengers only
# Categorical features: Survived, Sex, and Embarked. 

# Ordinal features: Pclass.

# Continuous features: Age, Fare. 

# Discrete: SibSp, Parch.

# Alpha-numeric: Ticket
train_df[train_df['Embarked'].isnull()]
train_df.describe()
train_df.describe(include=['O'])
#train_df.groupby('Ticket').size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)

train_df.groupby('Ticket').filter(lambda x: len(x)>1).sort_values(by=['Ticket'], ascending=False)
corr = train_df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.groupby('Survived').size()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

cleanup_nums = {"Sex":     {"male": 1, "female": 2},

                "Embarked": {"C": 1, "Q": 2, "S": 3}}

train_df.replace(cleanup_nums, inplace=True)

test_df.replace(cleanup_nums, inplace=True)
fill_NaN = Imputer(missing_values=np.nan, strategy='median', axis=1)

imputed_train_df = pd.DataFrame(fill_NaN.fit_transform(train_df))

imputed_train_df.columns = train_df.columns

imputed_train_df.index = train_df.index



imputed_test_df = pd.DataFrame(fill_NaN.fit_transform(test_df))

imputed_test_df.columns = test_df.columns

imputed_test_df.index = test_df.index
X_train = imputed_train_df.drop("Survived", axis=1)

Y_train = imputed_train_df["Survived"]

X_test  = imputed_test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

scores = cross_val_score(logreg, X_train, Y_train, cv=10)

acc_log = scores.mean()

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)

#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)





#acc_log
# Support Vector Machines



svc = SVC(C=10)



scores = cross_val_score(svc, X_train, Y_train, cv=10)

acc_svc = scores.mean()



print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)

#acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

#acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(knn, X_train, Y_train, cv=10)

acc_knn = scores.mean()

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Decision Tree



decision_tree = DecisionTreeClassifier()

scores = cross_val_score(decision_tree, X_train, Y_train, cv=10)

acc_decision_tree = scores.mean()

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

#acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(random_forest, X_train, Y_train, cv=10)

acc_random_forest = scores.mean()

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

#random_forest.score(X_train, Y_train)

#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#acc_random_forest
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN',

              'Decision Tree', 'Random Forest'],

    'Score': [acc_log, acc_svc, acc_knn, acc_decision_tree, 

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })
#submission.to_csv('../output/submission.csv', index=False)

submission.to_csv('submission.csv', index = False)
#submission_file = pd.read_csv('submission.csv')