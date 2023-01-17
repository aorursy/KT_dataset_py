# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype = {"Age": np.float64}, )



train.head()



#print("\n\nSummary statistics of training data")

#print(train.describe())



# Any results you write to the current directory are saved as output.
train.info()

print("-------------------------------")

test.info()
#Drop Unnecessary columns

train = train.drop(["PassengerId", "Name", "Ticket"], axis=1)

test = test.drop(["Name", "Ticket"], axis=1)



train.Embarked = train.Embarked.fillna(train.Embarked.mode()[0])

train.Embarked.describe()

train.Embarked.isnull().sum()
#Plotting Survival rate based on Embarked status

sns.factorplot('Embarked', 'Survived', data = train, size = 4, aspect = 1, kind="point") 
fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(10,5))

sns.countplot(x="Embarked", data=train, ax=axis1)

sns.countplot(x="Survived", hue="Embarked", data=train, order=[1,0],ax=axis2)

embarked_avg = train[["Embarked", "Survived"]].groupby("Embarked", as_index=False).mean()

sns.barplot(x="Embarked", y="Survived", data=embarked_avg, ax=axis3)
#Create dummy variables from Embarked



train_embarked_dummies = pd.get_dummies(train.Embarked)

test_embarked_dummies = pd.get_dummies(test.Embarked)



#Join dummy variables to test set and remove Embarked

train = train.join(train_embarked_dummies)

test = test.join(test_embarked_dummies)



train = train.drop("Embarked", axis=1)

test = test.drop("Embarked", axis=1)
#Fare value missing in test set

test["Fare"] = test.Fare.fillna(test.Fare.median())



survived_fare = train["Fare"][train.Survived == 1]

not_survived_fare = train["Fare"][train.Survived == 0]



fare_mean = pd.DataFrame([not_survived_fare.mean(), survived_fare.mean()])

fare_std = pd.DataFrame([not_survived_fare.std(), survived_fare.std()])



fare_mean.index.name = "Survived"

fare_mean.plot(yerr = fare_std, kind="bar", legend=False)



plt.figure()

train.Fare.plot(kind="hist",bins=100,xlim=(0,50))
#Age

age_mean_train = train.Age.mean()

age_std_train = train.Age.std()

age_na_train = train.Age.isnull().sum()

print([age_mean_train, age_std_train, age_na_train])



age_mean_test = test.Age.mean()

age_std_test = test.Age.std()

age_na_test = test.Age.isnull().sum()

print([age_mean_test, age_std_test, age_na_test])



rand_age_train = np.random.randint(age_mean_train - age_std_train, age_mean_train + age_std_train, size = age_na_train)

rand_age_test = np.random.randint(age_mean_test - age_std_test, age_mean_test + age_std_test, size = age_na_test)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10,5) ) 

#Plot old age values

train.Age.dropna().astype(int).hist(bins=70,ax=axis1)



#Replace NaN values with RNG values

train["Age"][np.isnan(train["Age"])] = rand_age_train

test["Age"][np.isnan(test["Age"])] = rand_age_test



#Convert

train.Age = train.Age.astype(int)

test.Age = test.Age.astype(int)



train.Age.hist(bins=70,ax=axis2)
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



plt.figure(figsize=(15,4))

survival_age = train[["Age", "Survived"]].groupby("Age",as_index=False).mean()

sns.pointplot(x="Age", y="Survived", data=survival_age)
#Drop Cabin because there are too many NaN values to define a proper impact

train = train.drop("Cabin", axis=1)

test = test.drop("Cabin", axis=1)
#Family



# Instead of having separate variables indicating Parch and SibSp,

# we can combine the two to see the total family members onboard



train["Family"] = train.Parch + train.SibSp

test["Family"] = test.Parch + test.SibSp

train.info()



train["Family_flag"] =0

train["Family_flag"].loc[train["Family"] > 0] = 1

train["Family_flag"].loc[train["Family"] == 0] = 0



test["Family_flag"] = 0

test["Family_flag"].loc[test["Family"] > 0] = 1

test["Family_flag"].loc[test["Family"] == 0] = 0



family_survival = train[["Family_flag", "Survived"]].groupby("Family_flag",as_index=False).mean()



fig, (axis1, axis2) = plt.subplots(1,2, figsize=(10,5))

sns.barplot(x="Family_flag", y = "Survived", order=[1,0],data=family_survival,ax=axis1)

axis1.set_xticklabels(["With Family", "Alone"])



sns.countplot(x="Family_flag", order=[1,0], data=train, ax=axis2)



train = train.drop(["Parch", "SibSp"], axis=1)

test = test.drop(["Parch", "SibSp"], axis=1)
train["Sex"].unique()

sex_flag = {"male" : 0, "female" : 1}

train["Sex"] = train["Sex"].map(sex_flag)

test["Sex"] = test["Sex"].map(sex_flag)
sex_survival = train[["Sex","Survived"]].groupby("Sex",as_index=False).mean()

ax = sns.barplot(x="Sex", y="Survived",data=sex_survival,order=[0,1])

ax.set(xticklabels=["Male","Female"])
class_survival = train[["Pclass", "Survived"]].groupby("Pclass", as_index=False).mean()



sns.barplot(x="Pclass", y="Survived", data=class_survival, order=[1,2,3])
train.info()

test.info()
from sklearn import model_selection as ms



X = train.drop(["Survived", "Family_flag"], axis=1)

Y = train.Survived

X_test = test.drop(["PassengerId","Family_flag"],axis=1)

X_train, X_cv, Y_train, Y_cv = ms.train_test_split(X, Y, test_size=0.25)

                                                

#print(X_train); print(Y_train.shape)
from sklearn import linear_model as lm



X_train, X_cv, Y_train, Y_cv = ms.train_test_split(X, Y, test_size=0.25)



logreg = lm.LogisticRegression(C=1,penalty='l1')

logreg = logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



print("Log Reg")

print(logreg.score(X_train,Y_train))

print(logreg.score(X_cv,Y_cv))



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier (min_samples_split=30, n_estimators=100)

rf = rf.fit(X_train,Y_train)



print("Random Forest")

print(rf.score(X_train,Y_train))

print(rf.score(X_cv,Y_cv))



coeff_df = pd.DataFrame(X_train.columns)

coeff_df.columns = ["Features"]

coeff_df["Logistic"] = logreg.coef_[0]

coeff_df["Forest"] = rf.feature_importances_



print(coeff_df)
Y_pred = rf.predict(X_test)

submission = pd.DataFrame({"PassengerId" : test.PassengerId, "Survived" : Y_pred })

submission.to_csv("titanic_predictions.csv", index=False)