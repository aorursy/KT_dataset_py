# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from scipy import stats

from scipy.stats import norm

from pandas.plotting import parallel_coordinates

%matplotlib inline
# Train Data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.columns
# Test Data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.columns
testId = test_data["PassengerId"]
# First few row values

train_data.head()
train_data.dtypes
train_data.drop(['PassengerId','Ticket'], axis=1, inplace = True)
train_data.Name.head() #train_data['Name'].head()
# Getting title from Name

train_title = [i.split(",")[1].split(".")[0].strip() for i in train_data["Name"]]

train_data["Title"] = pd.Series(train_title)

train_data["Title"].head()
# Getting title from Name

test_title = [i.split(",")[1].split(".")[0].strip() for i in test_data["Name"]]

test_data["Title"] = pd.Series(test_title)

test_data["Title"].head()
plt.figure(figsize = (10, 5))

g = sns.countplot(x="Title",data=train_data)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 

train_data["Title"] = train_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_data["Title"] = train_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

train_data["Title"] = train_data["Title"].astype(int)  # used to change dtype of title count
# Convert to categorical values Title 

test_data["Title"] = test_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_data["Title"] = test_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

test_data["Title"] = test_data["Title"].astype(int)  # used to change dtype of title count
plt.figure(figsize = (10, 5))

g = sns.countplot(x="Title",data=train_data)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Create a family size descriptor from SibSp and Parch

train_data["Family_size"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data["Family_size"] = test_data["SibSp"] + test_data["Parch"] + 1
train_data.isnull().sum()
#Fill Embarked nan values of dataset set with 'C' most frequent value

train_data["Embarked"] = train_data["Embarked"].fillna("C")

test_data["Embarked"] = test_data["Embarked"].fillna("C")



#complete missing fare with median

train_data['Fare'].fillna(train_data['Fare'].median(), inplace = True)

test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)



## Assigning all the null values as "N"

train_data.Cabin.fillna("N", inplace=True)

test_data.Cabin.fillna("N", inplace=True)
# group by Sex, Pclass, and Title 

grouped = train_data.groupby(['Sex','Pclass', 'Title'])  

# view the median Age by the grouped features 

grouped.Age.median()

# apply the grouped median value on the Age NaN

train_data.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
# group by Sex, Pclass, and Title 

grouped = test_data.groupby(['Sex','Pclass', 'Title'])  

# view the median Age by the grouped features 

grouped.Age.median()

# apply the grouped median value on the Age NaN

test_data.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
train_data.isnull().sum()
train_data['survived_dead'] = train_data['Survived'].apply(lambda x : 'Survived' if x == 1 else 'Dead')
sns.clustermap(data = train_data.corr().abs(),annot=True, fmt = ".2f", cmap = 'Reds')
plt.figure(figsize = (10, 5))

sns.countplot('survived_dead', data = train_data)
plt.figure(figsize = (10, 5))

sns.countplot( train_data['Sex'],data = train_data, hue = 'survived_dead', palette='coolwarm')
plt.figure(figsize = (10, 5))

sns.countplot( train_data['Pclass'],data = train_data, hue = 'survived_dead')
plt.figure(figsize = (10, 5))

sns.barplot(x = 'Pclass', y = 'Fare', data = train_data)
plt.figure(figsize = (10, 5))

sns.pointplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train_data)
plt.figure(figsize = (10, 5))

sns.barplot(x  = 'Embarked', y = 'Fare', data = train_data)
g = sns.FacetGrid(train_data, hue='Survived')

g.map(sns.kdeplot, "Age",shade=True)
sns.catplot(x="Embarked", y="Survived", hue="Sex",

            col="Pclass", kind = 'bar',data=train_data, palette = "rainbow")
sns.catplot(x='SibSp', y='Survived',hue = 'Sex',data=train_data, kind='bar')


sns.catplot(x='Parch', y='Survived',hue = 'Sex',data=train_data, kind='point')
g= sns.FacetGrid(data = train_data, row = 'Sex', col = 'Pclass', hue = 'survived_dead')

g.map(sns.kdeplot, 'Age', alpha = .75, shade = True)

plt.legend()
categoricals = train_data.select_dtypes(exclude=[np.number])

categoricals.describe()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



lbl = LabelEncoder() 

lbl.fit(list(train_data['Embarked'].values)) 

train_data['Embarked'] = lbl.transform(list(train_data['Embarked'].values))

lbl.fit(list(test_data['Embarked'].values)) 

test_data['Embarked'] = lbl.transform(list(test_data['Embarked'].values))
def encode(x): return 1 if x == 'female' else 0

train_data['enc_sex'] = train_data.Sex.apply(encode)

test_data['enc_sex'] = test_data.Sex.apply(encode)
train_data["has_cabin"] = [0 if i == 'N'else 1 for i in train_data.Cabin]

test_data["has_cabin"] = [0 if i == 'N'else 1 for i in test_data.Cabin]
from collections import Counter

# Outlier detection 



def detect_outliers(train_data, n, features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(train_data[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(train_data[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = train_data[(train_data[col] < Q1 - outlier_step) | (train_data[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train_data, 2, ["Age", "SibSp", "Parch", "Fare"])
train_data.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers

train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
data = train_data.select_dtypes(include=[np.number]).interpolate().dropna()
# Featuring the X-train, y_train

y_train = train_data["Survived"]



X_train = data.drop(labels = ["Survived"],axis = 1)
test_data = test_data.select_dtypes(include=[np.number]).interpolate().dropna()

test_data = test_data[X_train.columns]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)



test_data = sc.transform(test_data)
# Cross validate model with Kfold stratified cross val

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

kfold = StratifiedKFold(n_splits=10)
#ExtraTrees 

from sklearn.ensemble import ExtraTreesClassifier

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth":  [n for n in range(9, 14)],  

              "max_features": [1, 3, 10],

              "min_samples_split": [n for n in range(4, 11)],

              "min_samples_leaf": [n for n in range(2, 5)],

              "bootstrap": [False],

              "n_estimators" :[n for n in range(10, 60, 10)],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train,y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier()







## Search grid for optimal parameters

rf_param_grid = {"max_depth":  [n for n in range(9, 14)],  

              "max_features": [1, 3, 10],

              "min_samples_split": [n for n in range(4, 11)],

              "min_samples_leaf": [n for n in range(2, 5)],

              "bootstrap": [False],

              "n_estimators" :[n for n in range(10, 60, 10)],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Adaboost

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[30],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train,y_train)



ada_best = gsadaDTC.best_estimator_



gsadaDTC.best_score_
### SVC classifier

from sklearn.svm import SVC



SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
# Gradient boosting tunning

from sklearn.ensemble import GradientBoostingClassifier



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [n for n in range(10, 60, 10)],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth':  [n for n in range(9, 14)],  

              'min_samples_leaf': [n for n in range(2, 5)],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
from sklearn.ensemble import VotingClassifier



votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('svm',SVMC_best),

('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, y_train)
test_Survived = pd.Series(votingC.predict(test_data), name="Survived")



Submission = pd.concat([testId,test_Survived], axis=1)

Submission.to_csv("submission.csv",index=False)