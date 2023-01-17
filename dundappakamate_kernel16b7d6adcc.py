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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# This is new implementation with data claning, pattern studying and applying ML algorithims.

# https://www.kaggle.com/startupsci/titanic-data-science-solutions



# Good Kernel to refer.

#https://www.kaggle.com/ydalat/titanic-a-step-by-step-intro-to-machine-learning



# Data Analysis

import pandas as pd

import numpy as np



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt





# ML agorithims

# Accquire data

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_df,test_df]



# Describing data to find types like categorical, numerical, errors like comma, open bracket etc.

print( train_df.columns.values)

train_df.head()



# quick check at tail end

train_df.tail()



# dataytpes of each column

#train_df.info()

#print('_'*40)

# To view basic statistical data.

train_df.describe()

train_df.describe(include = ['O'])



train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



grid_1 = sns.FacetGrid(train_df,col='Survived')

grid_1.map(plt.hist,'Age',bins=20)



grid_2 = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')

grid_2.map(plt.hist,'Age', bins=20)



grid_3 = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid_3.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid_3.add_legend()



grid_4 = sns.FacetGrid(train_df, row='Embarked',col='Survived',  size=2.2, aspect=1.6)

grid_4.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid_4.add_legend()



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_df.head()  

pd.crosstab(train_df['Title'], train_df['Sex'])





for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

              'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()

    

    



    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare": 5}

for dataset in combine:

        dataset['Title'] = dataset['Title'].map(title_mapping)

        dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

test_df.head()



train_df = train_df.drop(['Name','PassengerId'], axis=1)

test_df = test_df.drop(['Name','PassengerId'], axis=1)

combine = [train_df, test_df]



#Convert Sex column from categorical to numerical.

for dataset in combine:

    dataset['Sex'] =  dataset['Sex'].map({"male":0,"female":1}).astype(int)



    train_df.head()

    

# start estimating and completing features with missing or null values

guess_ages = np.zeros((2,3))

guess_ages

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.tail()



# age banding and numbering into shot values.





for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()









# Age n class

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



# Work on  Family / alnone factor



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()





# Work on Embarked column

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)





#Convert Catgo to numerical feature.

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()



#Adding Missing fare value



test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace=True)

test_df.head()







#Convert the Fare feature to ordinal values based on the FareBand.

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



#train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)



test_df.head(10)
# After cleaning, populating, adding range, final datasets

X_train = train_df.drop("Survived",axis = 1)

Y_train = train_df["Survived"]



X_test = test_df.copy()

#Y_test  = ? need to predict this.

X_train.shape, Y_train.shape, X_test.shape

# Random Forest

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 7],

              "min_samples_split": [2, 3, 7],

              "min_samples_leaf": [1, 3, 7],

              "bootstrap": [False],

              "n_estimators" :[300,600],

              "criterion": ["gini"]}

random_forest = RandomForestClassifier(n_estimators=100,max_depth=20)



random_forest = GridSearchCV(random_forest,param_grid = rf_param_grid,  scoring="accuracy", n_jobs= 4, verbose = 1)





random_forest.fit(X_train, Y_train)

Y_pred_1 = random_forest.predict(X_test)



# Best score

random_forest_best = random_forest.best_estimator_

random_forest.best_score_



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Knn model

from sklearn.neighbors import KNeighborsClassifier #KNN

knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 

                           weights='uniform')

knn.fit(X_train, Y_train)

Y_pred_1 = knn.predict(X_test)

knn.score(X_train, Y_train)

knn = round(knn.score(X_train, Y_train) * 100, 2)

knn
# XGBClassifier

from xgboost import XGBClassifier

XGB = XGBClassifier(n_estimators= 100,random_state=1,max_depth=7,

                       learning_rate = 0.1,gamma=10)



XGB.fit(X_train, Y_train)

Y_pred_1 = XGB.predict(X_test)

XGB.score(X_train, Y_train)

acc_XGB = round(XGB.score(X_train, Y_train) * 100, 2)

acc_XGB

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=0)

decision_tree.fit(X_train, Y_train)

Y_pred_1 = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree

# SVC classifier

from sklearn.svm import SVC

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1,10,50,100,200,300, 1000]}

svc=SVC()



svc = GridSearchCV(SVMC,param_grid = svc_param_grid, scoring="accuracy", n_jobs= 4, verbose = 1)

svc.fit(X_train,Y_train)

Y_pred_1 = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



SVMC_best = svc.best_estimator_

# Best score

svc.best_score_
# Ensembling, Voting classifier

from sklearn.ensemble import VotingClassifier



VotingPredictor = VotingClassifier(estimators=[('SVMC', SVMC_best), ('random_forest', random_forest_best)], voting='soft', n_jobs=4)



VotingPredictor = VotingPredictor.fit(X_train, Y_train)

VotingPredictor_predictions = VotingPredictor.predict(X_test)

test_Survived = pd.Series(VotingPredictor_predictions, name="Survived")

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': VotingPredictor_predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")