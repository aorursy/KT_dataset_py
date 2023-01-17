# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_c

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
from sklearn.model_selection import train_test_split

#Read Train and Test Data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



y=train_data["Survived"]
train_data.info()

test_data.info()
print("The shape of train_data:", train_data.shape)

print("The shape of test_data:", test_data.shape)
train_data.head()
train_data.tail()
columns = train_data.columns

for col in columns:

    print("unique values in {} column is: {}". format(col, train_data[col].value_counts().size))
print("Total number of male passengers:")

print(train_data.loc[train_data.Sex=="male"].Sex.size)

print("Total number of female passengers:")

print(train_data.loc[train_data.Sex=="female"].Sex.size)
print("The percentage of survived with respect to Sex:")

print(100 * train_data.groupby("Sex").Survived.mean())
print("The percentage of survived with respect to Pclass:")

print(100 * train_data.groupby("Pclass").Survived.mean())
print("The percentage of survived with respect to Age:")

print(100 * train_data.groupby("Age").Survived.mean())
g = sns.FacetGrid(col="Survived", data=train_data, height = 2, aspect=3)

g.map(sns.distplot, "Age", kde=False, bins=80)
def cut_age(df, cut_values, label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age"]=pd.cut(df["Age"], bins=cut_values, labels=label_names)

    return df

    

cut_values=[-1, 0, 3, 12, 19, 35, 60, 80]

label_names=["Missing", "Infants", "Children", "Teenagers", "Young Adults", "Middle-Age Adults", "Seniors"]

train_data=cut_age(train_data, cut_values, label_names)

test_data=cut_age(test_data, cut_values, label_names)
sns.catplot(x="Age", row="Survived", kind="count", height=3, aspect=4, data=train_data)
print(100 * train_data.groupby("Age").Survived.mean())
print("The percentage of survived with respect to SibSp:")

print(100 * train_data.groupby("SibSp").Survived.mean())
print("The percentage of survived with respect to Parch:")

print(100 * train_data.groupby("Parch").Survived.mean())
def Cr_fam_membs(df):

    df["FamMembs"]= df["Parch"] + df["SibSp"]

    df=df.drop(["SibSp", "Parch"], axis=1)

    return df



train_data=Cr_fam_membs(train_data)

test_data=Cr_fam_membs(test_data)
print(100 * train_data.groupby("FamMembs").Survived.mean())
train_data["FamMembs"].unique()
test_data["FamMembs"].unique()
train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "IsAlone" if s==0 else s)

train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Small family" if (s==1 or s==2 or s==3) else s)

train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Meduim family" if (s==4 or s==5 or s==6) else s)

train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Large family" if (s==7 or s==10) else s)
test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "IsAlone" if s==0 else s)

test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Small family" if (s==1 or s==2 or s==3) else s)

test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Meduim family" if (s==4 or s==5 or s==6) else s)

test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Large family" if (s==7 or s==10) else s)
train_data["FamMembs"].value_counts()
test_data["FamMembs"].value_counts()
print("The percentage of survived with respect to Fam_membs:")

print(100 * train_data.groupby("FamMembs").Survived.mean())
sns.catplot(x="FamMembs", row="Survived", kind="count", height=3, aspect=4, data=train_data)
print("The percentage of survived with respect to Embarked:")

print(100 * train_data.groupby("Embarked").Survived.mean())
train_data["Embarked"]=train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

test_data["Embarked"]=test_data["Embarked"].fillna(test_data["Embarked"].mode()[0])
g = sns.FacetGrid(col="Survived", data=train_data, height = 2, aspect=3)

g.map(sns.distplot, "Fare", kde=False, bins=100)
sns.catplot(x="Pclass", y="Fare", kind="bar", data=train_data)
bins=np.arange(0, 600, 50)

g=sns.FacetGrid(row="Pclass", data=train_data, height = 3, aspect=5)

g.map(sns.distplot, "Fare", kde=False, bins=bins, color="b")
bins=np.arange(0, 600, 50)

g=sns.FacetGrid(col="Embarked", data=train_data, height = 3, aspect=2)

g.map(sns.distplot, "Fare", kde=False, bins=bins, color="b")
sns.catplot(x="Pclass", hue="Embarked", kind="count", data=train_data)
#Check number of passengers who embarked at each port

print(train_data.loc[train_data["Embarked"]=="S"].PassengerId.value_counts().sum())

print(train_data.loc[train_data["Embarked"]=="Q"].PassengerId.value_counts().sum())

print(train_data.loc[train_data["Embarked"]=="C"].PassengerId.value_counts().sum())
test_data["Fare"] = test_data["Fare"].fillna(-1)
train_data["Fare"].describe()
test_data["Fare"].describe()
def qcut_fare(df, q, labels):

    df["Fare"]=pd.qcut(df["Fare"], q, labels=labels)

    return df



labels=["range1", "range2", "range3", "range4"]

train_data=qcut_fare(train_data, 4, labels)

test_data=qcut_fare(test_data, 4, labels)
sns.catplot(x="Fare", data=train_data, kind="count", height=2, aspect=3)
sns.catplot(x="Fare", data=test_data, kind="count", height=2, aspect=3)
train_data["Name"]
train_data["Name"]=train_data["Name"].apply(lambda s: s.split(', ')[1].split('.')[0])

test_data["Name"]=test_data["Name"].apply(lambda s: s.split(', ')[1].split('.')[0])
train_data["Name"].unique()
test_data["Name"].unique()
train_data["Name"].value_counts()
test_data["Name"].value_counts()
train_data["Name"]=train_data["Name"].replace(["Ms", "Mlle"], "Miss")

train_data["Name"]=train_data["Name"].replace(["Sir"], "Mr")

train_data["Name"]=train_data["Name"].replace(["Mme"], "Mrs")

train_data["Name"]=train_data["Name"].replace(["Dr", "Rev", "Col", "Major", "Capt", "Master", 

                                             "Lady", "the Countess", "Don", "Dona", "Jonkheer"], "Other")
test_data["Name"]=test_data["Name"].replace(["Ms", "Mlle"], "Miss")

test_data["Name"]=test_data["Name"].replace(["Sir"], "Mr")

test_data["Name"]=test_data["Name"].replace(["Mme"], "Mrs")

test_data["Name"]=test_data["Name"].replace(["Dr", "Rev", "Col", "Major", "Capt", "Master", 

                                             "Lady", "the Countess", "Don", "Dona", "Jonkheer"], "Other")
train_data["Name"].unique()
train_data["Name"].value_counts()
test_data["Name"].value_counts()
sns.catplot(x="Name", hue="Survived", kind="count", data=train_data)
train_data.Cabin.describe()
train_data.Cabin.unique()
train_data["Cabin"]=train_data["Cabin"].fillna("Unknown")
test_data["Cabin"]=test_data["Cabin"].fillna("Unknown")
train_data["Cabin"].unique()
test_data["Cabin"].unique()
train_data["Deck"]=train_data["Cabin"].str.replace("([0-9\s])+","")
test_data["Deck"]=test_data["Cabin"].str.replace("([0-9\s])+","")
test_data["Deck"].value_counts()
train_data["Deck"].value_counts()
def total_cabins(row):

    if row.Deck == "Unknown":

        row["TotalCab"] = 0

    elif len(row.Deck) > 1:

        row["TotalCab"] = len(row.Deck)

    else:

        row["TotalCab"] = 1

    return row



train_data=train_data.apply(total_cabins, axis=1)

test_data=test_data.apply(total_cabins,axis=1)

        
train_data["TotalCab"].value_counts()
test_data["TotalCab"].value_counts()
train_data["Deck"]=train_data["Deck"].apply(lambda s: s[0] if s != "Unknown" else s)
test_data["Deck"]=test_data["Deck"].apply(lambda s: s[0] if s != "Unknown" else s)
test_data["Deck"].value_counts()
train_data["Deck"].value_counts()
train_data=train_data.drop(["Survived", "Cabin", "Ticket"], axis=1)

test_data=test_data.drop(["Cabin", "Ticket"], axis=1)
train_data.info()
test_data.info()


from sklearn.preprocessing import OneHotEncoder



OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)



features = ["Pclass", "Name", "Sex", "Age", "Fare", "Embarked", "FamMembs", "Deck", "TotalCab"]

OHE_train_cols = pd.DataFrame(OHE.fit_transform(train_data[features]))

OHE_test_cols = pd.DataFrame(OHE.transform(test_data[features]))



OHE_train_cols.index = train_data.index

OHE_test_cols.index = test_data.index



num_train=train_data.drop(features, axis=1)

num_test=test_data.drop(features, axis=1)



train_data = pd.concat([num_train, OHE_train_cols], axis=1)

test_data = pd.concat([num_test, OHE_test_cols], axis=1)

print(train_data.shape, test_data.shape)
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

xgb=XGBClassifier(objective='reg:logistic')



params={

    'n_estimators': [200, 500, 1000],

    'learning_rate': [0.01, 0.05, 0.1],

    'max_depth': [5, 7, 9],

    'colsample_bytree': [ 0.4, 0.6, 0.8],

    'subsample': [0.8, 0.9, 1],

    'gamma': [0, 0.5, 1]

}



clf=RandomizedSearchCV(xgb, param_distributions=params, n_iter=50, n_jobs=-1, verbose=1)

clf.fit(train_data, y)



score=clf.best_score_

params=clf.best_params_

print("Best score: ",score)

print("Best parameters: ", params)
final_predictions = clf.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")