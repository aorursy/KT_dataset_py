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
import random

import time

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



from subprocess import check_output

print(check_output(["ls", "/kaggle/input/titanic"]).decode("utf8"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



sns.set(style='white', context='notebook', palette='deep')
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

IDtest=test["PassengerId"]
# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
# view the outlier rows

train.loc[Outliers_to_drop]
# drop outliers 

train=train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
# join train and test datasets to obtain same features during categorical conversion

train_len=len(train)

dataset=pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# Fill empty and NA with NaN

dataset=dataset.fillna(np.nan)

dataset.isnull().sum()
train.info()

train.isnull().sum()
# summarize data

train.describe()
# Correlation matrix between numerical values (SibSp, Parch, Age and Fare Values) and Survived

g=sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
# Explore SibSp vs Survived

g=sns.factorplot(x='SibSp', y='Survived', data=train, kind='bar', size=6, palette='muted')

g.despine(left=True)

g=g.set_ylabels("Survival Probability")
# Explore Parch vs Survival

g=sns.factorplot(x='Parch', y='Survived', data=train, kind='bar', size=6, palette='muted')

g.despine(left=True)

g=g.set_ylabels("Survival Probability")
# Explore Age vs Survived

g=sns.FacetGrid(train, col='Survived')

g=g.map(sns.distplot, "Age")
g=sns.kdeplot(train['Age'][(train['Survived']==0) & (train['Age'].notnull())], color='Red', shade=True)

g=sns.kdeplot(train['Age'][(train['Survived']==1) & (train['Age'].notnull())], ax=g, color='Blue', shade=True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g=g.legend(["Not Survived", "Survived"])
# Fill missing Fare value with the median value

dataset["Fare"]=dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution

g=sns.distplot(dataset["Fare"], color="r", label="Skewness: %.2f"%(dataset["Fare"].skew()))

g=g.legend(loc="best")
# Apply log to Fare to reduce skewness of the distribution

dataset['Fare']=dataset['Fare'].map(lambda i: np.log(i) if i>0 else 0)



g=sns.distplot(dataset['Fare'], color='b', label='Skewness: %.2f'%(dataset['Fare'].skew()))

g=g.legend(loc='best')
# Explore Sex vs Survived

g=sns.barplot(x='Sex', y='Survived', data=train)

g=g.set_ylabel("Survival Probability")
# Explore Pclass vs Survived

g=sns.factorplot(x='Pclass', y='Survived', data=train, kind='bar', size=6, palette='muted')

g.despine(left=True)

g=g.set_ylabels("Survival Probability")
# Explore Pclass vs Survived by Sex

g=sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train, kind="bar", size=6, palette="muted")

g.despine(left=True)

g=g.set_ylabels("Survival Probability")
# Fill Embarked nan values of dataset set with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
#Explore Embarked vs Survived

g=sns.factorplot(x="Embarked", y="Survived", data=train, size=6, kind="bar", palette="muted")

g.despine(left=True)

g=g.set_ylabels("Survival Probability")
# Explore Pclass vs Embarked

g=sns.factorplot('Pclass', col='Embarked', data=train, kind='count', size=6, palette='muted')

g.despine(left=True)

g=g.set_ylabels("Count")
# Explore Age vs Sex, Parch, Pclass, SibSp

g=sns.factorplot(y='Age', x='Sex', data=dataset, kind='box')

g=sns.factorplot(y='Age', x='Sex', hue='Pclass', data=dataset, kind='box')

g=sns.factorplot(y='Age', x='Parch', data=dataset, kind='box')

g=sns.factorplot(y='Age', x='SibSp', data=dataset, kind='box')
# convert Sex into categorical value: 0 for male and 1 for female

dataset['Sex']=dataset['Sex'].map({'male':0, 'female':1})



g=sns.heatmap(dataset[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), cmap='BrBG', annot=True)
# Filling missing value of Age



## Fill Age with median age of similar rows according to Pclass, Parch, SibSp



# Index of NaN Age rows

index_NaN_Age=list(dataset['Age'][dataset['Age'].isnull()].index)



for i in index_NaN_Age:

    age_med=dataset['Age'].median()

    age_pred=dataset['Age'][((dataset['SibSp']==dataset.iloc[i]['SibSp']) & 

                             (dataset['Parch']==dataset.iloc[i]['Parch']) &

                             (dataset['Pclass']==dataset.iloc[i]['Pclass']))].median()

    if not np.isnan(age_pred):

        dataset['Age'].iloc[i]=age_pred

    else:

        dataset['Age'].iloc[i]=age_med
g=sns.factorplot(y='Age', x='Survived', data=train, kind='violin')
dataset['Title']=dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())



def replace_title(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'the Countess', 'Lady', 'Dona']:

        return 'Rare'

    elif title in ['Mlle', 'Ms', 'Mrs', 'Miss', 'Mme']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Female':

            return 'Miss'

        else:

            return 'Mr'

    else:

        return title



dataset['Title']=dataset.apply(replace_title, axis=1)

dataset['Title'].unique()

dataset['Title']=dataset['Title'].map({"Mr":0, "Miss": 1, "Master":2, "Rare": 3})

dataset['Title']=dataset['Title'].astype(int)
g=sns.countplot(dataset['Title'])

g=g.set_xticklabels(['Mr', 'Miss', 'Master', 'Rare'])
g=sns.factorplot(x='Title', y='Survived', data=dataset, kind='bar', size=6, palette='muted')

g=g.set_xticklabels(['Mr', 'Miss', 'Master', 'Rare'])

g=g.set_ylabels("Survival Probability")
# Drop Name variable

dataset.drop(labels=['Name'], axis=1, inplace=True)
# Create Family Size feature

dataset['Fsize']=dataset['SibSp']+dataset['Parch']+1



g=sns.factorplot(x='Fsize', y='Survived', data=dataset)

g=g.set_ylabels("Survival Probability")
# New features basis the family size

dataset['Single']=dataset['Fsize'].map(lambda s: 1 if s==1 else 0)

dataset['SmallF']=dataset['Fsize'].map(lambda s: 1 if s==2 else 0)

dataset['MediumF']=dataset['Fsize'].map(lambda s: 1 if 3<=s<=4 else 0)

dataset['LargeF']=dataset['Fsize'].map(lambda s: 1 if s>=5 else 0)
g=sns.factorplot(x='Single', y='Survived', data=dataset, size=6, kind='bar', palette='muted')

g=g.set_ylabels('Survival Probability')

g=sns.factorplot(x='SmallF', y='Survived', data=dataset, size=6, kind='bar', palette='muted')

g=g.set_ylabels('Survival Probability')

g=sns.factorplot(x='MediumF', y='Survived', data=dataset, size=6, kind='bar', palette='muted')

g=g.set_ylabels('Survival Probability')

g=sns.factorplot(x='LargeF', y='Survived', data=dataset, size=6, kind='bar', palette='muted')

g=g.set_ylabels('Survival Probability')
# convert to indicator values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset['Cabin'].describe()

dataset['Cabin'].isnull().sum()

dataset['Cabin'][dataset['Cabin'].notnull()].head()

dataset['Cabin']=pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])

g=sns.countplot(dataset['Cabin'], order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
g=sns.factorplot(x='Cabin', y='Survived', data=dataset, kind='bar', size=6,

                 order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])

g=g.set_ylabels("Survival Probability")
dataset=pd.get_dummies(dataset, columns=['Cabin'], prefix='Cabin')

dataset.head()
Ticket=[]

for i in list(dataset.Ticket):

    if not i.isdigit():

        Ticket.append(i.replace(".", " ").replace("/", " ").strip().split(" ")[0])

    else:

        Ticket.append("X")



dataset['Ticket']=Ticket

dataset['Ticket'].head()
dataset=pd.get_dummies(dataset, columns=['Ticket'], prefix='T')
# create categorical values for Pclass

dataset['Pclass']=dataset['Pclass'].astype("category")

dataset=pd.get_dummies(dataset, columns=['Pclass'], prefix='PC')
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
dataset.head()
# separate train and test datasets

train=dataset[:train_len]

test=dataset[train_len:]

test.drop(labels=['Survived'], axis=1, inplace=True)
# separate train features and label

train['Survived']=train['Survived'].astype(int)

Y_train=train['Survived']

X_train=train.drop(labels=['Survived'], axis=1)
# Cross validate model with Kfold stratified cross validation

kfold=StratifiedKFold(n_splits=10)



# Modeling Step - test different algorithms 

random_state=2

classifiers=[]

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,

                                      learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state=random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results=[]

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))



cv_means=[]

cv_std=[]

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res=pd.DataFrame({"CrossValMeans": cv_means, "CrossValErrors":cv_std, 

                     "Algorithm":["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees", 

                                  "GradientBoosting", "MultipleLayerPerceptron", "KNeighbors", 

                                  "LogisticRegression", "LinearDiscriminantAnalysis"]})



g=sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g=g.set_title("Cross Validation Scores")
# AdaBoost

DTC=DecisionTreeClassifier()

adaDTC=AdaBoostClassifier(DTC, random_state=7)

ada_param_grid={"base_estimator__criterion":["gini", "entropy"], 

                "base_estimator__splitter": ["best", "random"],

                "algorithm": ["SAMME", "SAMME.R"],

                "n_estimators": [1,2],

                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC=GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsadaDTC.fit(X_train, Y_train)

ada_best=gsadaDTC.best_estimator_
gsadaDTC.best_score_
# ExtraTrees

ExtC=ExtraTreesClassifier()

ex_param_grid={"max_depth": [None],

               "max_features": [1, 3, 10],

               "min_samples_split": [2, 3, 10],

               "min_samples_leaf": [1, 3, 10],

               "bootstrap": [False],

               "n_estimators": [100, 300],

               "criterion": ["gini"]}

gsExtC=GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsExtC.fit(X_train, Y_train)

ExtC_best=gsExtC.best_estimator_

gsExtC.best_score_
# Random Forest Classifier

RFC=RandomForestClassifier()

rf_param_grid={"max_depth": [None],

               "max_features": [1, 3, 10],

               "min_samples_split": [2, 3, 10],

               "min_samples_leaf": [1, 3, 10],

               "bootstrap": [False],

               "n_estimators": [100, 300],

               "criterion": ["gini"]}

gsRFC=GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsRFC.fit(X_train, Y_train)

RFC_best=gsRFC.best_estimator_

gsRFC.best_score_
# Gradient Boosting

GBC=GradientBoostingClassifier()

gb_param_grid={"loss": ["deviance"],

               "n_estimators": [100, 200, 300],

               "learning_rate": [0.1, 0.05, 0.01],

               "max_depth": [4, 8],

               "min_samples_leaf":[100, 150],

               "max_features": [0.3, 0.1]}

gsGBC=GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsGBC.fit(X_train, Y_train)

GBC_best=gsGBC.best_estimator_

gsGBC.best_score_
# SVC Classifier

SVC=SVC(probability=True)

svc_param_grid={"kernel": ["rbf"],

                "gamma":[0.001, 0.001, 0.01, 0.1, 1],

                "C":[1, 10, 50, 100, 200, 300, 1000]}

gsSVC=GridSearchCV(SVC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsSVC.fit(X_train, Y_train)

SVC_best=gsSVC.best_estimator_

gsSVC.best_score_