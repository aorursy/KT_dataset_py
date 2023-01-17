# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load The Dataset

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
# Check first 5 rows of the data

df_train.head()
# Check the columns

df_train.columns
# Check the shape of the data

df_train.shape
# Check all data (mean, min,max etc) 

df_train.describe()
# Check the data types of all columns in dataset

df_train.info()
# Check how many passengers on Ship

print("Total Passenger : ",len(df_train["PassengerId"]))
# Check how many survived or how many not survived

survived = df_train[df_train["Survived"]==1]

not_survived = df_train[df_train["Survived"]==0]
print(f"Total Passenger Who Survived : {len(survived)}")

print(f"Total Passener Who Not Survived : {len(not_survived)}")
print("Survived % = ",1*len(survived)/len(df_train)*100)

print("Not Survived % = ",1*len(not_survived)/len(df_train)*100)
# Check which class has more passengers and check how much this impact on survived or not_survived

df_train["Pclass"].value_counts().plot(kind="bar")
plt.figure(figsize=(6,12))

plt.subplot(211)

sns.countplot(x="Pclass",data=df_train)

plt.subplot(212)

sns.countplot(x="Pclass",data=df_train,hue="Survived",palette="summer")
## Check how much sex can impact on survived or not_survived

plt.figure(figsize=(6,12))

plt.subplot(211)

sns.countplot(x="Sex",data=df_train)

plt.subplot(212)

sns.countplot(x="Sex",data=df_train,hue="Survived",palette="winter")
plt.figure(figsize=(8,12))

plt.subplot(211)

sns.countplot(x="SibSp",data=df_train)

plt.subplot(212)

sns.countplot(x="SibSp",data=df_train,hue="Survived",palette="winter_d")
plt.figure(figsize=(8,12))

plt.subplot(211)

sns.countplot(x="Parch",data=df_train)

plt.subplot(212)

sns.countplot(x="Parch",data=df_train,hue="Survived",palette="winter_d")
plt.figure(figsize=(8,12))

plt.subplot(211)

sns.countplot(x="Embarked",data=df_train)

plt.subplot(212)

sns.countplot(x="Embarked",data=df_train,hue="Survived",palette="winter_d")
# Check the histogram of Age

df_train["Age"].hist(bins=40)
# Check the histogram of Fare

df_train["Fare"].hist(bins=40)
# Check how many null values have in data

plt.figure(figsize=(10,8))

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap="Blues")
## Those column who is True are null values columns and second print showing null values columns

print(df_train.isna().any())

print(df_train.loc[:,df_train.isna().any()])
# Now drop unussual columns like Cabin,Name,PassengerId,Ticket because this are not so useful..

df_train.drop(["Ticket","PassengerId","Name","Cabin"],axis=1,inplace=True)

df_test.drop(["Ticket","PassengerId","Name","Cabin"],axis=1,inplace=True)
# Check the first 5 rows of data for verify that column successfull dropped

df_train.head()
# Groupby Age With Sex And Check the mean of male age and female age for fill nan(null) values

df_train.groupby("Sex").agg({"Age":["mean"]})
df_test.groupby("Sex").agg({"Age":["mean"]})
# Create def fucntion for filling nan values of age

def fill_age(data):

    age = data[0]

    sex = data[1]

    if pd.isnull(age):

        if sex is "male":

            return 30

        else:

            return 27

    else:

        return age
def fill_age_test(data):

    age = data[0]

    sex = data[1]

    if pd.isnull(age):

        if sex is "male":

            return 30

        else:

            return 30

    else:

        return age
# Now fill the nan values of age with mean of male and female

df_train["Age"] = df_train[["Age","Sex"]].apply(fill_age,axis=1)

df_test["Age"] = df_test[["Age","Sex"]].apply(fill_age_test,axis=1)
# Now verify nan values filled

plt.figure(figsize=(10,8))

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=True,cmap="Blues")
df_train.isna().any()
# Now check histogram of Embarked and check which Embarked have more passenger then fill then nan value

df_train["Embarked"].hist()
df_test["Embarked"].hist()
## Embarked S Have More Passengers So That's why we choose Emabarked S For fill nan data

print(df_train["Embarked"].value_counts())

print(df_train["Embarked"].value_counts().idxmax())
df_train["Embarked"].replace(np.nan,"S",inplace=True)

df_test["Embarked"].replace(np.nan,"S",inplace=True)
# Now verify nan Data filled or not

plt.figure(figsize=(10,8))

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=True,cmap="Blues")
# So We Successfully removed all missing value from data and drop unussual columns

df_train.isna().any()
df_test.isna().any()
fare = df_test["Fare"].astype("float64").mean(axis=0)

fare
df_test["Fare"].replace(np.nan,fare,inplace=True)
df_test.isna().any()
df_test.dtypes
# Now check dtypes of all columns

df_train.dtypes
# So we have to convert object = int or float because sklearn not take objects only take numeric values like int,float etc
# Create dummy variable for Sex

sex_variable = pd.get_dummies(df_train["Sex"],drop_first=True)

sex_variable
# Now conat the sex_vaiable with df_train and drop the Sex columns

df_train = pd.concat([df_train,sex_variable],axis=1)

df_train.drop("Sex",axis=1,inplace=True)
# Verify the data correct or not

df_train.head()
sex_variable = pd.get_dummies(df_test["Sex"],drop_first=True)
df_test = pd.concat([df_test,sex_variable],axis=1)

df_test.drop("Sex",axis=1,inplace=True)
# Create dummy variable for Embarked

embarked_var = pd.get_dummies(df_train["Embarked"],drop_first=True)

embarked_var
# Now conat the embarked_var with df_train and drop the Embarked columns

df_train = pd.concat([df_train,embarked_var],axis=1)

df_train.drop("Embarked",axis=1,inplace=True)
# Verify the data correct or not

df_train.head()
# Now Change the column name of Q and S with Embarked_Q and Embarked_S

df_train = df_train.rename(columns={"Q":"Embarked_Q","S":"Embarked_S"})
# Now Check the data all are correct or not

df_train.head()
embarked_var = pd.get_dummies(df_test["Embarked"],drop_first=True)
df_test = pd.concat([df_test,embarked_var],axis=1)

df_test.drop("Embarked",axis=1,inplace=True)
df_test = df_test.rename(columns={"Q":"Embarked_Q","S":"Embarked_S"})
# Now check the dtypes of all the columns all are good or not

df_train.dtypes
df_test.dtypes
X = df_train.drop("Survived",axis=1).values
y = df_train["Survived"].values
print(X.shape)

print(y.shape)
# Feature Scaling

sc = StandardScaler()

X = sc.fit_transform(X)

df_test = sc.fit_transform(df_test)
X
df_test
# Model Training
lin_model = LogisticRegression(random_state=10)

lin_model.fit(X,y)
Survived = lin_model.predict(df_test)
Survived
new_df = pd.read_csv("/kaggle/input/titanic/test.csv")
new_df.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"],axis=1,inplace=True)
Survived = pd.DataFrame(Survived)
Survived
new_df = pd.concat([new_df,Survived],axis=1)
new_df = new_df.rename(columns={0:"Survived"})

new_df.columns
new_df.to_csv("titanic_submissions_1",index=False)