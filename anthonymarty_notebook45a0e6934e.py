# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# importing pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotli, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC # support vector machine

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# get data

titanic_df = pd.read_csv("../input/train.csv", dtype = {"Age": np.float64},)

test_df = pd.read_csv("../input/test.csv", dtype ={"Age": np.float64},)



# preview of data

titanic_df.head()
titanic_df.info()

test_df.info()
# drop unnecessary info

#titanic_df.drop(['PassengerId','Name','Ticket'],axis = 1)

#test_df.drop([('Name','Ticket')], axis = 1)



# Embarked



# fill the only 2 missing values of titanic_df with "S"

#titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



# plot

sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



#

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

sns.countplot(x = 'Survived', hue = 'Embarked', data = titanic_df, order = [1,0], ax = axis2)



# group by embarked and get the meanfor survived passengers for each value in embarked

embark_perc = titanic_df[["Embarked","Survived"]].groupby(["Embarked"],as_index = False).mean()

sns.barplot(x = 'Embarked', y = "Survived",data=embark_perc,order=['S','C','Q'],ax=axis3)
#Age

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(9,4))

axis1.set_title('Original age values - Titanic')

axis2.set_title('New age values - Titanic')



# get avg, std and nber of NaN values in titanic_df

avg_age_titanic = titanic_df["Age"].mean()

std_age_titanic = titanic_df["Age"].std()

nan_age_titanic = titanic_df["Age"].isnull().sum()



# get avg, std and nber of NaN values in test_df

avg_age_test = test_df["Age"].mean()

std_age_test = test_df["Age"].std()

nan_age_test = test_df["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

# https://docs.python.org/2/library/random.html

rand_1 = np.random.randint(avg_age_titanic - std_age_titanic, avg_age_titanic + std_age_titanic, 

                       size = nan_age_titanic)

rand_2 = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test, 

                       size = nan_age_test)



# plot original age values

# drop all null values and convert to int

titanic_df["Age"].dropna().astype(int).hist(bins = 70, ax=axis1)



# fill na values with values rand generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2



# convert float to int

titanic_df["Age"] = titanic_df["Age"].astype(int)

test_df["Age"] = test_df["Age"].astype(int)



# plot new age values

titanic_df["Age"].hist(bins = 70, ax = axis2)
# family

titanic_df["Family"] = titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df["Family"].loc[titanic_df["Family"] > 0 ] = 1

titanic_df["Family"].loc[titanic_df["Family"] == 0 ] = 0



test_df["Family"] = test_df["Parch"] + test_df["SibSp"]

test_df["Family"].loc[test_df["Family"] > 0 ] = 1

test_df["Family"].loc[test_df["Family"] == 0 ] = 0



# droping parch adn sibsp

titanic_df = titanic_df.drop(["Parch","SibSp"],axis=1)

test_df = test_df.drop(["Parch","SibSp"],axis=1)



fig, (axis1,axis2) = plt.subplots(1,2, sharex = True, figsize = (10,5))

sns.countplot(x = "Family", data = titanic_df, order = [1,0], ax = axis1)



# average of survived for those who had/didn't have any family member

#family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

family_perc = titanic_df[["Survived","Family"]].groupby(["Family"], as_index = False).mean()

sns.barplot(x = "Family", y = "Survived", data = family_perc, order = [1,0], ax = axis2)



axis1.set_xticklabels(["With family","Alone"], rotation = 0)
# sex

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person, axis = 1)

test_df['Person'] = test_df[['Age','Sex']].apply(get_person, axis = 1)



#  no need to use sex column because we created person

# about dop: http://chrisalbon.com/python/pandas_dropping_column_and_rows.html

# inplace : To delete the column without having to reassign df you can do:

titanic_df.drop(["Sex"], axis=1, inplace = True)

test_df.drop(["Sex"], axis=1, inplace = True)



person_dummies_titanic = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ["Child","Female","Male"]

person_dummies_titanic.drop(["Male"], axis = 1, inplace = True)



person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ["Child","Female","Male"]

person_dummies_test.drop(["Male"], axis = 1, inplace = True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df = test_df.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2, figsize=(10,5))

sns.countplot(x = 'Person', data = titanic_df, ax = axis1)



# average of survived for each Person(male, female, or child)

person_perc = titanic_df[["Person","Survived"]].groupby(["Person"], as_index = False).mean()

sns.barplot(x = 'Person', y ='Survived', data = person_perc, ax = axis2, order = ["Male","Female","Child"])



titanic_df.drop(["Person"], axis = 1, inplace = True)

test_df.drop(["Person"], axis = 1, inplace = True)
# influence of Pclass

sns.factorplot( x = "Pclass", y = "Survived", order = [1,2,3], data = titanic_df, size = 5)