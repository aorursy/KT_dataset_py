# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from pandas.tools.plotting import parallel_coordinates
%matplotlib inline
train_data = pd.read_csv("../input/train.csv")
train_data.columns
test = pd.read_csv("../input/test.csv")
IDtest = test["PassengerId"]

train_data.head()
train_data.dtypes
train_data.drop(['PassengerId','Ticket'], axis=1, inplace = True)

train_data.isnull().sum()
 #complete missing age with mean
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)

#Fill Embarked nan values of dataset set with 'S' most frequent value
train_data["Embarked"] = train_data["Embarked"].fillna("C")
test["Embarked"] = test["Embarked"].fillna("C")

#complete missing fare with median
train_data['Fare'].fillna(train_data['Fare'].median(), inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)

## Assigning all the null values as "N"
train_data.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)
train_data.isnull().sum()
train_data["Name"].head()
# Get Title from Name
train_title = [i.split(",")[1].split(".")[0].strip() for i in train_data["Name"]]
train_data["Title"] = pd.Series(train_title)
train_data["Title"].head()
# Get Title from Name
test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
test["Title"] = pd.Series(test_title)
test["Title"].head()
g = sns.countplot(x="Title",data=train_data)
g = plt.setp(g.get_xticklabels(), rotation=45) 

# Convert to categorical values Title 
train_data["Title"] = train_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data["Title"] = train_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train_data["Title"] = train_data["Title"].astype(int)
# Convert to categorical values Title 
test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
test["Title"] = test["Title"].astype(int)
g = sns.countplot(x="Title",data=train_data)
g = plt.setp(g.get_xticklabels(), rotation=45) 

# Create a family size descriptor from SibSp and Parch
train_data["Family_size"] = train_data["SibSp"] + train_data["Parch"] + 1
test["Family_size"] = test["SibSp"] + test["Parch"] + 1

# Create new feature of family size
train_data['Single'] = train_data['Family_size'].map(lambda s: 1 if s == 1 else 0)
train_data['Small_family'] = train_data['Family_size'].map(lambda s: 1 if  s == 2  else 0)
train_data['Med_family'] = train_data['Family_size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train_data['Large_family'] = train_data['Family_size'].map(lambda s: 1 if s >= 5 else 0)

# Create new feature of family size
test['Single'] = test['Family_size'].map(lambda s: 1 if s == 1 else 0)
test['Small_family'] = test['Family_size'].map(lambda s: 1 if  s == 2  else 0)
test['Med_family'] = test['Family_size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['Large_family'] = test['Family_size'].map(lambda s: 1 if s >= 5 else 0)
train_data['survived_dead'] = train_data['Survived'].apply(lambda x : 'Survived' if x == 1 else 'Dead')
sns.clustermap(data = train_data.corr().abs(),annot=True, fmt = ".2f", cmap = 'Blues')
sns.countplot('survived_dead', data = train_data)
sns.countplot( train_data['Sex'],data = train_data, hue = 'survived_dead', palette='coolwarm')
sns.countplot( train_data['Pclass'],data = train_data, hue = 'survived_dead')
sns.barplot(x = 'Pclass', y = 'Fare', data = train_data)
sns.pointplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train_data);
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
#train_data['Male'] = train_data['Sex'].map(lambda s: 1 if s == 'male' else 0)
#train_data['female'] = train_data['Sex'].map(lambda s: 1 if  s == 'female'  else 0)
#
#test['Male'] = test['Sex'].map(lambda s: 1 if s == 'male' else 0)
#test['female'] = test['Sex'].map(lambda s: 1 if  s == 'female'  else 0)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lbl = LabelEncoder() 
lbl.fit(list(train_data['Embarked'].values)) 
train_data['Embarked'] = lbl.transform(list(train_data['Embarked'].values))
lbl.fit(list(test['Embarked'].values)) 
test['Embarked'] = lbl.transform(list(test['Embarked'].values))
train_data['FareBin'] = pd.qcut(train_data['Fare'], 4)
train_data['AgeBin'] = pd.cut(train_data['Age'].astype(int), 5)

test['FareBin'] = pd.qcut(test['Fare'], 4)
test['AgeBin'] = pd.cut(test['Age'].astype(int), 5)
train_data['AgeBin_Code'] = lbl.fit_transform(train_data['AgeBin'])
train_data['FareBin_Code'] = lbl.fit_transform(train_data['FareBin'])
    
test['AgeBin_Code'] = lbl.fit_transform(test['AgeBin'])
test['FareBin_Code'] = lbl.fit_transform(test['FareBin'])
def encode(x): return 1 if x == 'female' else 0
train_data['enc_sex'] = train_data.Sex.apply(encode)
test['enc_sex'] = test.Sex.apply(encode)
train_data["has_cabin"] = [0 if i == 'N'else 1 for i in train_data.Cabin]
test["has_cabin"] = [0 if i == 'N'else 1 for i in test.Cabin]
from collections import Counter
# Outlier detection 

def detect_outliers(train_data,n,features):
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
Outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])

train_data.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers
train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
data = train_data.select_dtypes(include=[np.number]).interpolate().dropna()

y_train = train_data["Survived"]

X_train = data.drop(labels = ["Survived"],axis = 1)
test = test.select_dtypes(include=[np.number]).interpolate().dropna()
test = test[X_train.columns]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

test = sc.transform(test)
# Cross validate model with Kfold stratified cross val
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)
#ExtraTrees 
from sklearn.ensemble import ExtraTreesClassifier
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [4],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
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
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
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
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
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

test_Survived = pd.Series(votingC.predict(test), name="Survived")

Submission = pd.concat([IDtest,test_Survived],axis=1)
Submission.to_csv("submission.csv",index=False)
