# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

# There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks

# The 'whitegrid' theme is similar, but it is better suited to plots with heavy data elements:

# %matplotlib inline activates the inline backend and calls images as static pngs.

sns.set_style('whitegrid')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
#get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype = {"Age" : np.float64}, sep=",")

test_df    = pd.read_csv("../input/test.csv", dtype = {"Age" : np.float64}, sep=",")



#preview date

titanic_df.head()
# to show columns of the dataset acombined with the data types.

titanic_df.info()

print ("-----------------------------------------")

test_df.info()

#titanic_df.describe()

#print ("-----------------------------------------")

#test_df.describe()
# drop unnecessary columns, these columns won't be useful in analysis and prediction

#Note: axis=1 denotes that we are referring to a column

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# Embarked

# only in titanic_df, fill the two missing values with the most common value, which is "S"

# we can find the most common value using pandas.DataFrame.mode function, it returns a sorted dataframe 

# of the most common values in each column in case 'axis=0'

most_common = titanic_df.mode(axis=0, numeric_only=False)

print (most_common)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



# plot

# seaborn.factorplot(): Draw a categorical plot onto a FacetGrid.

# aspect = 3, Aspect ratio of each facet, so that 'aspect * size' gives the width of each facet in inches.

# size = 4, Height (in inches) of each facet. 

# if you don't specify a plt type it will be 'point' by default.

sns.factorplot('Embarked','Survived', data = titanic_df, size = 4, aspect = 3)

#subplot(nrows, ncols, plot_number)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize = (15,5))



# order: Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.

# hue:  hue variable should be used for the most important comparisons

# kind : type of plot --> {point, bar, count, box, violin, strip}

#sns.factorplot('Embarked', data = titanic_df, kind = 'count', order = ['S','C','Q'], ax = axis1)

#sns.factorplot('Survived', hue = 'Embarked', data = titanic_df, kind = 'count', order = [1,0], ax = axis2)

   ##### OR you can use countplot() instead of using factorplot( kind = 'count')

sns.countplot(x= 'Embarked', data = titanic_df, ax = axis1)

sns.countplot(x = 'Survived', hue = 'Embarked', data = titanic_df, order = [1,0], ax = axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked ['S','C','Q']

embark_perc = titanic_df[["Embarked","Survived"]].groupby(['Embarked'], as_index = False).mean()

sns.barplot(x = "Embarked", y = "Survived", data = embark_perc, order = ['S','C','Q'], ax= axis3)



#embark_perc # uncomment to take a look at the mean of survived at each port S, C, and Q



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable --> as it has the least mean 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.



# get_dummies(): Convert categorical variable into dummy/indicator variables

embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis = 1, inplace = True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis = 1, inplace=True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis = 1, inplace = True)

test_df.drop(['Embarked'], axis = 1, inplace = True)
#Fare



# only for test_df, since there is a missing "Fare" values

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)



# convert from float to int ------------------->this snippet constantly is raising error

#titanic_df['Fare'] = int(titanic_df['Fare'])

#test_df['Fare'] = int(test_df['Fare'])

#titanic_df['Fare'] = titanic_df['Fare'].astype(int)

#test_df['Fare']    = test_df['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = titanic_df['Fare'][titanic_df['Survived'] == 0]

fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]



# get average,standard deviation for fare survived/not survived passengers

#class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

# data : numpy ndarray (structured or homogeneous), dict, or DataFrame

#Dict can contain Series, arrays, constants, or list-like objects

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])



#plot

# kind='hist': is histogram kind plot

# bins: If an integer, divide the counts in the specified number of bins, and color the hexagons accordingly,

# which means that for the larger bar in the histogram color_pixels = [count/frequency = 350]/[binsNumber = 100] = 3.5

# xlim( (xmin, xmax) )  # set the xlim to xmin, xmax

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



# index:  Index to use for resulting frame. Will default to np.arange(n) if no indexing information part of input data and no index provided

average_fare.index.names = std_fare.index.names = ["Survived"]

# yerr: will be used to generate errorbar(s) on the bar chart

average_fare.plot(yerr =std_fare, kind='bar', legend=False)
# Age

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# axis3.set_title('Original Age values - Test')

# axis4.set_title('New Age values - Test')



# get average, std, and number of NaN values in titanic_df

average_age_titanic   = titanic_df["Age"].mean()

std_age_titanic       = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test = test_df["Age"].mean()

std_age_test = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std) for titanic_df & test_df

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,

size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test,

size = count_nan_age_test)



## plot original Age values

# NOTE: drop all null values, and convert to int

titanic_df["Age"].dropna().astype(int).hist(bins = 70, ax= axis1)

#test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2



# convert from float to int

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)



#plot new Age values

titanic_df["Age"].hist(bins = 70, ax = axis2)

#test_df["Age"].hist(bins = 70, ax = axis4)
# .... continue with plot Age column



# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(titanic_df, hue = "Survived", aspect = 4)

facet.map(sns.kdeplot,"Age", shade = True)

facet.set(xlim = (0, titanic_df["Age"].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_df[["Age","Survived"]].groupby(["Age"], as_index = False).mean()

sns.barplot(x = "Age", y = "Survived", data = average_age)
# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)
# Family



## Instead of having two columns Parch[Parent & child] & SibSp [Sibling & spouse], 

# we can have only one column represent if the passenger had any family member aboard or not,

# if passenger has family_members>0 put 1, if family_members == 0 put 0

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic_df["Family"] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df["Family"].loc[titanic_df["Family"] > 0]  = 1

titanic_df["Family"].loc[titanic_df["Family"] == 0] = 0



test_df["Family"] = test_df["Parch"] + test_df["SibSp"]

test_df["Family"].loc[test_df["Family"] > 0]  = 1

test_df["Family"].loc[test_df["Family"] == 0] = 0



# drop Sibsp & Parch columns

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



# Plot

# sharex = True, your subplots share the same x-axis

fig, (axis1, axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)



# average of survived for those who had/ didn't have any family members

family_perc = titanic_df[["Family","Survived"]].groupby(["Family"], as_index=False).mean()

sns.barplot(x="Family", y="Survived", data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# Sex



## As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child



def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



#pandas.DataFrame.apply, Applies function along input axis of DataFrame.

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person, axis = 1)

test_df['Person'] = test_df[['Age','Sex']].apply(get_person, axis = 1)



# No need to use Sex column since we created Person column

# inplace : bool, default False

# If True, means do operation inplace and return None

titanic_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers as we saw below

person_dummies_titanic = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df = test_df.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person', data = titanic_df, ax=axis1)



# average of survived for each Person(male, female, or child)

# as_index = False,is effectively “SQL-style” grouped output

person_perc = titanic_df[['Person','Survived']].groupby(['Person'], as_index = False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)
# Pclass



sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class1','Class2','Class3']

pclass_dummies_titanic.drop(['Class3'], axis=1, inplace=True)



pclass_dummies_test = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class1','Class2','Class3']

pclass_dummies_test.drop(['Class3'], axis=1, inplace=True)



titanic_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df = test_df.join(pclass_dummies_test)

# define trainig and testing sets

X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop(["PassengerId"],axis=1).copy()

test_df['Fare'] = test_df['Fare'].median(skipna=True)
# Logistic regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test) #Predict class labels for samples in X_test

logreg.score(X_train, Y_train)
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)



svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)