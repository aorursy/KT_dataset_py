# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
train.info()
test.info()
all = pd.concat([train, test], sort = False)

all.info()
#Fill Missing numbers with median

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())
all.info()
sns.catplot(x = 'Embarked', kind = 'count', data = all) #or all['Embarked'].value_counts()
all['Embarked'] = all['Embarked'].fillna('S')

all.info()
#Age

all.loc[ all['Age'] <= 16, 'Age'] = 0

all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1

all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2

all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3

all.loc[ all['Age'] > 64, 'Age'] = 4 
#Title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()
#Cabin

all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]

all['Cabin'].value_counts()
#Family Size & Alone 

all['Family_Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family_Size']==1, 'IsAlone'] = 1

all.head()
#Drop unwanted variables

all_1 = all.drop(['Name', 'Ticket'], axis = 1)

all_1.head()
all_dummies = pd.get_dummies(all_1, drop_first = True)

all_dummies.head()
all_train = all_dummies[all_dummies['Survived'].notna()]

all_train.info()
all_test = all_dummies[all_dummies['Survived'].isna()]

all_test.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1), 

                                                    all_train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

knn.score(X_test,y_test)
neighbors = np.arange(1,9)

test_accuracy = np.empty(len(neighbors))

train_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,y_train)

    test_accuracy[i] = knn.score(X_test,y_test)

    train_accuracy[i] =knn.score(X_train,y_train)

    

plt.plot(neighbors, test_accuracy, label='Test Accuracy')

plt.plot(neighbors, train_accuracy, label='Train Accuracy')

plt.legend()

plt.show()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver = 'liblinear')

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

predictions
X = all_train.drop(['PassengerId','Survived'],axis=1) 

y= all_train['Survived']
X.columns
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver = 'liblinear')

cv_results = cross_val_score(reg,X,y, cv = 8)

print(cv_results)

print(np.mean(cv_results))
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge



alpha_space = np.logspace(-4,0,50)

ridge_scores = []

ridge_scores_std = []

ridge = Ridge(normalize = True)

for alpha in alpha_space:

    ridge.alpha = alpha

    ridge_cv_scores = cross_val_score(ridge,X,y, cv=9)

    ridge_scores.append(np.mean(ridge_cv_scores))

    ridge_scores_std.append(np.std(ridge_cv_scores))

    

print(ridge_scores)



def display_plot(cv_scores, cv_scores_std):

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    ax.plot(alpha_space, cv_scores)



    std_error = cv_scores_std / np.sqrt(10)



    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

    ax.set_ylabel('CV Score +/- Std Error')

    ax.set_xlabel('Alpha')

    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

    ax.set_xlim([alpha_space[0], alpha_space[-1]])

    ax.set_xscale('log')

    plt.show()

    

display_plot(ridge_scores, ridge_scores_std)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge



ridge = Ridge(normalize = True, alpha = 0.2)

ridge_cv_scores = cross_val_score(ridge,X,y, cv=9)



print(np.mean(ridge_cv_scores))
ridge = Ridge(normalize = True, alpha = 0.1)

ridge_cv_scores = cross_val_score(ridge,X,y, cv=9)



print(np.mean(ridge_cv_scores))
ridge = Ridge(normalize = True, alpha = 0.3)

ridge_cv_scores = cross_val_score(ridge,X,y, cv=9)



print(np.mean(ridge_cv_scores))
ridge = Ridge(normalize = True, alpha = 0.17)

ridge_cv_scores = cross_val_score(ridge,X,y, cv=8)

ridge_coef = ridge.fit(X,y).coef_

print(np.mean(ridge_cv_scores))

print(ridge_coef)

names= X.columns



plt.plot(range(len(names)),ridge_coef)

plt.xticks(range(len(names)), names, rotation=90)

plt.show()
from sklearn.model_selection import GridSearchCV

alphas = np.array([.15, .16,.17, 0.18, .19])

ridge = Ridge(normalize = True)

ridge_cv = GridSearchCV(ridge,param_grid=dict(alpha=alphas),cv=8 )

ridge_cv.fit(X,y)

print(ridge_cv.best_params_)

print(ridge_cv.best_score_)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
all_test.head()
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
TestForPred.info()
t_pred = logmodel.predict(TestForPred).astype(int)
T_r_pred = ridge_cv.predict(TestForPred).astype(int)
PassengerId = all_test['PassengerId']
logSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })

logSub.head()
logSub['Survived'].sum()
ridgeSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':T_r_pred })

ridgeSub.head()
ridgeSub['Survived'].sum()
logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)
ridgeSub.to_csv("2_Logistics_Regression_Submission.csv", index = False)