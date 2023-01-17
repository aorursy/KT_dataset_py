import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

combine = pd.concat([train,test],ignore_index=True)
train.columns.values
train.head()
train.info()

print(" ")

test.info()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.figure(figsize=(16,6))

sns.heatmap(train.corr(),cmap="YlGnBu",square=True,linewidths=.5,center=0,linecolor="red")
# Grab title from passenger name



s = combine["Name"]

s = s.str.split(pat=",", expand = True)

s = s[1].str.split(pat=".", expand = True)

s= s[0]

combine["title"] = s
combine["title"].value_counts()
pd.crosstab(combine["Sex"],combine["title"] , rownames = ['Sex'], colnames = ['title'])
# Reassign mlle, ms, and mme accordingly

combine['title'] = combine['title'].str.replace('Mlle', 'Miss')

combine['title'] = combine['title'].str.replace('Ms', 'Miss')

combine['title'] = combine['title'].str.replace('Mme', 'Mrs')



# Ttitles with very low cell counts to be combined to "rare" level

rare_title = '|'.join(['Dona','Lady','the Countess','Capt','Col','Don', 

                'Dr','Major','Rev','Sir','Jonkheer'])



combine['title'] = combine['title'].str.replace(rare_title, 'raretitle')
pd.crosstab(combine["Sex"],combine["title"] , rownames = ['Sex'], colnames = ['title'])
# Finally, grab surname from passenger name

s = combine["Name"]

s = s.str.split(pat=",", expand = True)[0]

combine["surname"] = s
# Create a family size variable including the passenger themselves

combine["Fsize"] = combine["SibSp"] + combine["Parch"] + 1



# Create a family variable 

combine["Family"] = combine.surname + "_" + combine.Fsize.map(str) 
# Visualize the relationship between family size & survival

sns.countplot(x="Fsize", data=combine,hue="Survived").set(xlabel="Family Size", ylabel = "Count")
# Discretize family size

def family_size(row):

    if row['Fsize'] == 1:

        val = "singleton"

    elif row["Fsize"] > 4:

        val = "large"

    elif row['Fsize'] < 5 and row['Fsize'] > 1:

        val = "small"

    return val



combine['FsizeD'] = combine.apply(family_size, axis=1)
from statsmodels.graphics.mosaicplot import mosaic
# Show family size by survival using a mosaic plot

mosaic(combine, ['FsizeD', 'Survived'],title="Family Size by Survival")
# This variable appears to have a lot of missing values

combine["Cabin"][0:27]
# The first character is the deck. For example:



list(combine["Cabin"][1])
# Create a Deck variable. Get passenger deck A - F:



combine["Deck"] = combine["Cabin"].str[0]
combine.Embarked[combine.Embarked != combine.Embarked].index.values
# Passengers 62 and 830 are missing Embarkment

print(combine["Embarked"][61])

print(combine["Embarked"][829])
# Get rid of our missing passenger IDs

embarked_fare = combine[combine["PassengerId"] != 62]

embarked_fare = embarked_fare[embarked_fare["PassengerId"] != 830]



# Visualize embarkment, passenger class, & median fare

fig_dims = (10, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x="Embarked",y="Fare",hue="Pclass",ax=ax,data=embarked_fare)
# Since their fare was $80 for 1st class, they most likely embarked from 'C'

combine.loc[combine["PassengerId"] == 62,"Embarked"] = "C"

combine.loc[combine["PassengerId"] == 830,"Embarked"] = "C"
combine.loc[[1043]]
p1d = combine[((combine["Pclass"] == 3) & (combine["Embarked"] == "S"))]



sns.kdeplot(p1d["Fare"], shade=True).axvline(p1d['Fare'].median(), color="red", linestyle="--")
# Replace missing fare value with median fare for class/embarkment

combine["Fare"][1043] = p1d['Fare'].median()
# Show number of missing Age values

combine["Age"].isna().sum() 
# excluding certain less-than-useful variables:

ddf = combine.copy()

ddf.drop(['PassengerId','Name','Ticket','Cabin','Family','surname','Survived'], axis=1,inplace=True)



# Set a random seed

import random

random.seed(129)



# Import the imputation package and predict missing age values

from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent")

mice_output = imp.fit_transform(ddf)



# Assignt it to a pandas data frame and give the columns their original names

mice_output = pd.DataFrame(mice_output).copy()

mice_output.columns = ["Pclass","Sex","Age","SisSp","Parch","Fare","Embarked","title","Fsize","FsizeD","Deck"]
plt.hist(combine["Age"], bins=16)

plt.ylim(0, 400)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Age: Original Data')

print(plt.show())

print("")

plt.hist(mice_output["Age"], bins=16)

plt.ylim(0, 400)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Age: Mice Output')

print(plt.show())
# Replace Age variable from the mice model.

combine["Age"] = mice_output["Age"]



# Show new number of missing Age values

combine["Age"].isnull().sum()
# First we'll look at the relationship between age & survival

# I include Sex since we know (a priori) it's a significant predictor

bins = np.linspace(combine[0:890].Age.min(), combine.Age.max(), 25)

g = sns.FacetGrid(combine, col="Sex", hue="Survived", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
# Create the column child, and indicate whether child or adult

combine["IsAdult"] = np.where(combine['Age'] < 18, '0', '1')



# Show counts

pd.crosstab(combine["IsAdult"],combine["Survived"] 

            , rownames = ['IsAdult'], colnames = ['Survived'])
# Adding Mother variable

combine["IsMother"] = np.where((combine['Sex'] == "female") 

                             & (combine["Parch"] > 0) & (combine["Age"] > 18) 

                             & (combine["title"] != "Miss")

                             , '1', '0')



# Show counts

pd.crosstab(combine["IsMother"],combine["Survived"] 

            , rownames = ['IsMother'], colnames = ['Survived'])
# Drop the useless variables

X = combine.copy()

X.drop(["Cabin","Name","PassengerId","Ticket","surname"

           ,"Family","Deck"], axis=1,inplace=True)
# Label encoding the binary Sex variable

X['Sex'].replace(to_replace=['male','female'], value=[1,0],inplace=True)



# Preview data

X
# Creating a feature variable to keep original data intact

Feature = X.copy()



# One hot encoding

Feature = pd.get_dummies(Feature, columns=['Pclass', 'Embarked',"title","FsizeD"], drop_first=True)
Feature
X_train = Feature[0:891].copy()

X_train.drop(["Survived"], axis = 1,inplace=True)

X_test = Feature[891:1309].copy()

X_test.drop(["Survived"], axis = 1,inplace=True)

Y_train = train['Survived'].values
from sklearn import preprocessing

X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)

X_train[0:5]
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)

X_test[0:5]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( X_train, Y_train, test_size=0.3, random_state=4)
error_rate = []



for i in (200,300,500,750,1000,1500,2000):

    

    rfc = RandomForestClassifier(n_estimators=i)

    rfc.fit(x_train,y_train)

    pred_i = rfc.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot((200,300,500,750,1000,1500,2000),error_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",

        markersize=10)

plt.title("Error Rate vs n_estimators Value")

plt.xlabel("n_estimators")

plt.ylabel("Error Rate")
rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train,Y_train)
rfc_pred = rfc.predict(X_test)

rfc_pred = pd.DataFrame(rfc_pred)

rfc_pred["PassengerId"] = test["PassengerId"]

rfc_pred.columns = ['Survived', 'PassengerId']

rfc_pred = rfc_pred[['PassengerId', 'Survived']]

rfc_pred
rfc_pred.to_csv("titanic_pred.csv", index=False)
error_rate = []



for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,50),error_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",

        markersize=10)

plt.title("Error Rate vs K Value")

plt.xlabel("K")

plt.ylabel("Error Rate")
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,Y_train)

pred_KNN = knn.predict(X_test)

pred_KNN = pd.DataFrame(pred_KNN)

pred_KNN["PassengerId"] = test["PassengerId"]

pred_KNN.columns = ['Survived', 'PassengerId']

pred_KNN = pred_KNN[['PassengerId', 'Survived']]

pred_KNN
pred_KNN.to_csv("titanic_pred_knn.csv", index=False)
error_rate = []



for i in (0.001,0.01,0.1,1,2,3,4,5):

    

    LR = LogisticRegression(C=i, solver='liblinear').fit(x_train,y_train)

    pred_i = LR.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot((0.001,0.01,0.1,1,2,3,4,5),error_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",

        markersize=10)

plt.title("Error Rate vs C Value")

plt.xlabel("C")

plt.ylabel("Error Rate")
LR = LogisticRegression(C = 1,solver='liblinear').fit(X_train,Y_train)
# Prediction:

submission = LR.predict(X_test)
# Transforming it into the format that kaggle wants us to upload as:

submission = pd.DataFrame(submission)

submission["PassengerId"] = test["PassengerId"]

submission.columns = ['Survived', 'PassengerId']

submission = submission[['PassengerId', 'Survived']]

submission
# exporting as CSV

# submission.to_csv("submission.csv", index=False)