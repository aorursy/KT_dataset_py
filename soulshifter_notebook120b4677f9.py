# $ 0 U l $ h ! f T 3 r



# IMPORT BASIC LIBRARIES



import numpy as np



import pandas as pd

pd.set_option('display.height', 2000)

pd.set_option('display.max_rows', 2000)

pd.set_option('display.max_columns', 2000)

pd.set_option('display.width', 2000)

pd.set_option('display.max_colwidth', -1)



import matplotlib

from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')



import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# DATA LOAD

train_set = pd.read_csv("/home/vic/Desktop/Kaggle/Titanic/train.csv")

test_set = pd.read_csv("/home/vic/Desktop/Kaggle/Titanic/test.csv")



# DATA CHECK

"""print(train_set.describe())

print(train_set.describe(include=["object"]))

print(train_set.columns)

print(train_set.shape)

print(test_set.describe())

print(test_set.describe(include=["object"]))

print(test_set.columns)

print(test_set.shape)

print(train_set.head())"""





# SETTING TARGET

target = train_set.Survived



# DROPPING UNNECCESSARY FEATS

train_set.drop(["PassengerId","Survived","Ticket"],axis=1,inplace=True)

test_pas_id = test_set.PassengerId

test_set.drop(["PassengerId","Ticket"],axis=1,inplace=True)

# print(test_pas_id.head())



# NUMERIC FEATS AND CATEGORICAL FEATS

train_numfeats = train_set.select_dtypes(exclude=["object"]).columns

test_numfeats = test_set.select_dtypes(exclude=["object"]).columns

train_catfeats = train_set.select_dtypes(include=["object"]).columns

test_catfeats = test_set.select_dtypes(include=["object"]).columns





# ADDING AND DELETING FEATURES

train_set["Familysize"] = train_set["Parch"] + train_set["SibSp"]

train_set.drop(["SibSp","Parch"],axis=1,inplace=True)

test_set["Familysize"] = test_set["Parch"] + test_set["SibSp"]

test_set.drop(["SibSp","Parch"],axis=1,inplace=True)



# NUMERIC FEATS AND CATEGORICAL FEATS

train_numfeats = train_set.select_dtypes(exclude=["object"]).columns

test_numfeats = test_set.select_dtypes(exclude=["object"]).columns

train_catfeats = train_set.select_dtypes(include=["object"]).columns

test_catfeats = test_set.select_dtypes(include=["object"]).columns



# CHECKING NULL FEATS

# print(train_set.isnull().sum())

# print(test_set.isnull().sum())



# PLOT TO CHECK HOW TO FILL MISSING VALUES

# train_set["Age"].hist(bins=15,color="teal",alpha=0.8)

# sns.countplot(x="Embarked",data=train_set,palette="Set2")

# train_set["Fare"].hist(bins=15,color="teal",alpha=0.8)

# plt.show()





# DEALING WITH FEATS

train_set["Age"].fillna(train_set["Age"].median(),inplace=True)

train_set["Embarked"].fillna("S",inplace=True)

train_set.drop(["Cabin"],axis=1,inplace=True)

test_set["Age"].fillna(train_set["Age"].median(),inplace=True)

test_set["Embarked"].fillna("S",inplace=True)

test_set["Fare"].fillna(train_set["Fare"].median(),inplace=True)

test_set.drop(["Cabin"],axis=1,inplace=True)



# CHECKING FEATS AGAIN AFTER PROCESSING

# print(train_set.isnull().sum())

# print(test_set.isnull().sum())





# NUMERIC FEATS AND CATEGORICAL FEATS

train_numfeats = train_set.select_dtypes(exclude=["object"]).columns

test_numfeats = test_set.select_dtypes(exclude=["object"]).columns

train_catfeats = train_set.select_dtypes(include=["object"]).columns

test_catfeats = test_set.select_dtypes(include=["object"]).columns

print(train_numfeats)

print(train_catfeats)



# DEALING WITH NAMES

replacement = {

    'Don': 0,

    'Rev': 0,

    'Jonkheer': 0,

    'Capt': 0,

    'Mr': 1,

    'Dr': 2,

    'Col': 3,

    'Major': 3,

    'Master': 4,

    'Miss': 5,

    'Mrs': 6,

    'Mme': 7,

    'Ms': 7,

    'Mlle': 7,

    'Sir': 7,

    'Lady': 7,

    'the Countess': 7

}



train_set["Name"] = train_set["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())

# print(train_set["Name"].unique())

train_set["Name"] = train_set["Name"].apply(lambda x: replacement.get(x))

# print(train_set["Name"].head())

test_set["Name"] = test_set["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())

# print(test_set["Name"].unique())

test_set["Name"] = test_set["Name"].apply(lambda x: replacement.get(x))

# print(test_set["Name"].head())



# DEALING WITH SEX

replacement1 = {

    "male":1,

    "female":0

}



train_set["Sex"] = train_set["Sex"].apply(lambda x: replacement1.get(x))

test_set["Sex"] = test_set["Sex"].apply(lambda x: replacement1.get(x))

# print(train_set["Sex"].head())

# print(test_set["Sex"].head())



# DEALING WITH EMBARKED

# print(train_set["Embarked"].unique())

replacement2 = {

    "S":0,

    "C":1,

    "Q":2

}



train_set["Embarked"] = train_set["Embarked"].apply(lambda x: replacement2.get(x))

test_set["Embarked"] = test_set["Embarked"].apply(lambda x: replacement2.get(x))

# print(train_set["Embarked"].head())

# print(test_set["Embarked"].head())



# THERE IS NAN VALUE IN TEST SET

test_set["Name"].fillna(test_set["Name"].mode()[0],inplace=True)



# SCALING ATTRIBUTES

from sklearn.preprocessing import StandardScaler

train_set["Pclass"] = StandardScaler().fit_transform(train_set["Pclass"].values.reshape(-1,1))

test_set["Pclass"] = StandardScaler().fit_transform(test_set["Pclass"].values.reshape(-1,1))

train_set["Age"] = StandardScaler().fit_transform(train_set["Age"].values.reshape(-1,1))

test_set["Age"] = StandardScaler().fit_transform(test_set["Age"].values.reshape(-1,1))

train_set["Fare"] = StandardScaler().fit_transform(train_set["Fare"].values.reshape(-1,1))

test_set["Fare"] = StandardScaler().fit_transform(test_set["Fare"].values.reshape(-1,1))

train_set["Familysize"] = StandardScaler().fit_transform(train_set["Familysize"].values.reshape(-1,1))

test_set["Familysize"] = StandardScaler().fit_transform(test_set["Familysize"].values.reshape(-1,1))

train_set["Name"] = StandardScaler().fit_transform(train_set["Name"].values.reshape(-1,1))

test_set["Name"] = StandardScaler().fit_transform(test_set["Name"].values.reshape(-1,1))

train_set["Sex"] = StandardScaler().fit_transform(train_set["Sex"].values.reshape(-1,1))

test_set["Sex"] = StandardScaler().fit_transform(test_set["Sex"].values.reshape(-1,1))

train_set["Embarked"] = StandardScaler().fit_transform(train_set["Embarked"].values.reshape(-1,1))

test_set["Embarked"] = StandardScaler().fit_transform(test_set["Embarked"].values.reshape(-1,1))



# print(train_set.isnull().sum())

# print(test_set.isnull().sum())





# MODELLING

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

param_grid = {"n_estimators":[250],"max_depth":[8]}



clf = GridSearchCV(rf,param_grid=param_grid,cv=10,n_jobs=-1)

clf.fit(train_set,target)

print(clf.best_score_)

pred = clf.predict(test_set)





# SUBMISSION

submission = pd.DataFrame({"PassengerId":test_pas_id,"Survived":pred})

submission.to_csv("/home/vic/Desktop/Kaggle/Titanic/final2.csv",index=False)