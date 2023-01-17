# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn import metrics



#Train_data

train_data = pd.read_csv('../input/train.csv',)

train_data.head()

train_data.describe()







#Test_data

test_data = pd.read_csv('../input/test.csv',)

test_data.head()

test_data.describe()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Remove feature ticket from the DataFrame(rows,colums)

train_data.drop(['Ticket'],axis=1)
# Remove feature ticket from the test DataFrame(rows,colums)

test_data.drop(['Ticket'],axis=1)
sb.set(style="ticks")

train_data.info()
g = sb.lmplot(x="Age", y="PassengerId",ci=None,data=train_data, col="Survived",

    palette="muted",col_wrap=2,scatter_kws={"s": 100,"alpha":.5},

    line_kws={"lw":4,"alpha":0.5},hue="Survived",x_jitter=1.0,y_jitter=1.0,size=6)



# remove the top and right line in graph

sb.despine()

# Additional line to adjust some appearance issue

plt.subplots_adjust(top=0.9)



# Set the Title of the graph from here

g.fig.suptitle('Age vs. PassengerId', fontsize=10,color="b",alpha=0.5)



# Set the xlabel of the graph from here

g.set_xlabels("Age",size = 10,color="b",alpha=0.5)



# Set the ylabel of the graph from here

g.set_ylabels("PassengerId",size = 10,color="b",alpha=0.5)
# Plotting of Embarkment at different points and Survival

sb.factorplot(x="Embarked", data=train_data, kind="count",

                   palette="BuPu", hue='Survived', size=6, aspect=1.5)

# Plotting of different Sex and Survival

sb.factorplot(x="Sex", data=train_data, kind="count",

                   palette="BuPu", hue='Survived',size=6, aspect=1.5)

# Plotting for different Age and Survival

sb.factorplot(x="Age", data=train_data, kind="count",

                   palette="BuPu", hue='Survived',size=6, aspect=1.5)
# Plotting  Survival from Cabins

g = sb.factorplot("Survived", col="Cabin", col_wrap=4, data=train_data[train_data.Cabin.notnull()],kind="count", size=2.5, aspect=.8)
# Plotting  Survival from SibSp

g = sb.factorplot("Survived", col="SibSp", col_wrap=4, data=train_data[train_data.SibSp.notnull()],kind="count", size=2.5, aspect=.9)
# Plotting  Survival from Sex

g = sb.factorplot("Survived", col="Sex", col_wrap=4, data=train_data[train_data.Sex.notnull()],kind="count", size=2.5, aspect=.8)
# Plotting  Survival from Age

g = sb.factorplot("Survived", col="Age", col_wrap=4, data=train_data[train_data.Age.notnull()],kind="count", size=2.5, aspect=.8)
# Plotting  Survival from PassengerId according to their Sex

g = sb.factorplot(x="Sex", y="PassengerId", col="Survived",data=train_data, saturation=.5,kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate").set_xticklabels(["Men", "Women", "Children"]).set_titles("{col_name} {col_var}").set(ylim=(0, 1)).despine(left=True)) 
# Plotting  Embarked according to Age

g = sb.factorplot(x="Age", y="Embarked",hue="Sex", row="Pclass", data=train_data[train_data.Embarked.notnull()],orient="h", size=2, aspect=3.5, palette="Set3",kind="violin", split=True, cut=0, bw=.2)
# Replace values in the DataFrame which are not identified

def random_age():

    for age in train_data["Age"]:

            train_data.Age.fillna(0)

            sum_age = sum(train_data.Age.fillna(0))/train_data.index.size

            age = np.random.randint((sum_age)//1)

    return age



def random_age():

    for age in train_data["Fare"]:

            train_data.Fare.fillna(0)

            sum_fare = sum(train_data.Fare.fillna(0))/train_data.index.size

            fare = np.random.randint((sum_age)//1)

    return fare





def random_passengerId():

    for passengerId in train_data["PassengerId"]:

            passengerId = np.random.randint(sum(train_data["PassengerId"])/train_data.index.size)

    return passengerId



train_data["Age"][np.isnan(train_data["Age"])] = random_age()

train_data["PassengerId"][np.isnan(train_data["PassengerId"])] = random_passengerId()



# Transform Type Object into valid Type for the DataFrame 

train_data['Fare'] = train_data['Fare'].astype(int)



#Convert Sex with 0  for male and 1 for female

train_data.loc[train_data["Sex"] == "male", "Sex"] = 0

train_data.loc[train_data["Sex"] == "female", "Sex"] = 1



#Converting Embarked points(S,C,Q) with numbers

print(train_data["Embarked"].unique())

train_data["Embarked"] = train_data["Embarked"].fillna("S")

train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0

train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1

train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2



# Create a Pairplot

g = sb.pairplot(train_data,hue="Survived",palette="muted",size=5, 

    vars=["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"],kind='reg')



# To change the size of the scatterpoints in graph

g = g.map_offdiag(plt.scatter,  s=150, alpha=0.5)
# Making Predictions

# Firt try the LinearRegression

#instantiate

linreg = LogisticRegression()



# Selecting the Features and Target

def random_age():

    for age in train_data["Age"]:

            train_data.Age.fillna(0)

            sum_age = sum(train_data.Age.fillna(0))/train_data.index.size

            age = np.random.randint((sum_age)//1)

    return age

    

train_data["Age"][np.isnan(train_data["Age"])] = random_age()





#Convert Sex with 0  for male and 1 for female

train_data.loc[train_data["Sex"] == "male", "Sex"] = 0

train_data.loc[train_data["Sex"] == "female", "Sex"] = 1



#Converting Embarked points(S,C,Q) with numbers

print(train_data["Embarked"].unique())

train_data["Embarked"] = train_data["Embarked"].fillna("S")

train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0

train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1

train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2



train_data.Survived.add

#X_train = train_data[['Age']]

#train_data.head()

predictors=["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"]

X_train = train_data[predictors]



y_train = train_data['Survived']

#y_train = train_data['Survived']

print(X_train)

# fit the model to the training Data

linreg.fit(X_train, y_train)
# print the intercept and the coefficients

print(linreg.intercept_)

print(linreg.coef_)
# Making prediction on the testing Data test



# Selecting the Features and Target

def random_age():

    for age in test_data["Age"]:

            test_data.Age.fillna(0)

            sum_age = sum(test_data.Age.fillna(0))/test_data.index.size

            age = np.random.randint((sum_age)//1)

    return age



predictors = ["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"]

X_train = train_data[predictors]

y_train = train_data["Survived"]



train_data["Age"][np.isnan(train_data["Age"])] = random_age()



X_test = train_data[predictors]

y_pred = linreg.predict(X_test)



print(y_pred)



linreg.score(X_train,y_train)
# Using KNNearest Classifier

# instantiate

knn = KNeighborsClassifier(n_neighbors=6)

print(knn)



#Fit/Train the model

# Selecting the Features and Target

def random_age():

    for age in train_data["Age"]:

            train_data.Age.fillna(0)

            sum_age = sum(train_data.Age.fillna(0))/train_data.index.size

            age = np.random.randint((sum_age)//1)

    return age



predictors = ["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"]



#X_train = train_data.drop("Survived",axis=1)

Y_train = train_data["Survived"]

#X_test  = test_data.drop("PassengerId",axis=1).copy()

X_train = train_data[predictors]



train_data["Age"][np.isnan(train_data["Age"])] = random_age()



#train_data.Survived.add

#X_train = train_data[['Age']]

#train_data.head()

y_train = train_data['Survived']



knn.fit(X_train,y_train)

knn.score(X_train,y_train)
# Prediction

def random_age():

    for age in test_data["Age"]:

            test_data.Age.fillna(0)

            sum_age = sum(test_data.Age.fillna(0))/test_data.index.size

            age = np.random.randint((sum_age)//1)

    return age



def random_fare():

    for age in test_data["Fare"]:

            test_data.Fare.fillna(0)

            sum_fare = sum(test_data.fare.fillna(0))/test_data.index.size

            fare = np.random.randint((sum_fare)//1)

            test_data['Fare'] = test_data['Fare'].astype(int)

    return fare



# Transform Type Object into valid Type for the DataFrame 

#test_data['Fare'] = test_data['Fare'].astype(int)





#X_train = titanic_df.drop("Survived",axis=1)

#Y_train = titanic_df["Survived"]

#X_test  = test_df.drop("PassengerId",axis=1).copy()



#Convert Sex with 0  for male and 1 for female

#test_data.loc[test_data["Sex"] == "male", "Sex"] = 0

#test_data.loc[test_data["Sex"] == "female", "Sex"] = 1



#Converting Embarked points(S,C,Q) with numbers

#print(test_data["Embarked"].unique())

#test_data["Embarked"] = test_data["Embarked"].fillna("S")

#test_data.loc[test_data["Embarked"] == "S", "Embarked"] = 0

#test_data.loc[test_data["Embarked"] == "C", "Embarked"] = 1

#test_data.loc[test_data["Embarked"] == "Q", "Embarked"] = 2



#predictors = ["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"]

#X_test = test_data[predictors]

#test_data["Age"][np.isnan(test_data["Age"])] = random_age()



#X_train = train_data[predictors]

#y_test = test_data["Survived"]

#X_test = test_data[predictors]

#y_pred = knn.predict(X_test)



#score = metrics.accuracy_score(X_train,knn.predict(X_test))

#print("Accuracy:",score)



#print(y_pred)
def random_age():

    for age in train_data["Age"]:

            train_data.Age.fillna(0)

            sum_age = sum(train_data.Age.fillna(0))/train_data.index.size

            age = np.random.randint((sum_age)//1)

    return age



def random_fare():

    for age in test_data["Fare"]:

            test_data.Fare.fillna(0)

            sum_fare = sum(test_data.fare.fillna(0))/test_data.index.size

            fare = np.random.randint((sum_fare)//1)

            test_data['Fare'] = test_data['Fare'].astype(int)

    return fare



#predictors = ["Age","Sex", "PassengerId", "SibSp", "Parch", "Pclass", "Fare","Embarked"]





#logreg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

#logreg.fit(predictors, test_data["Survived"])



# Make predictions using the test set.

#predictions = logreg.predict(predictors)



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pandas.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('titanic.csv', index=False)