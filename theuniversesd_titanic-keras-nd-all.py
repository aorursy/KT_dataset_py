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
#This Librarys is to work with matrics

import pandas as pd

#This Librarys is to work with vectors

import numpy as np

#This Librarys is to work with visualization

import seaborn as sns

#to render the graphics

import matplotlib.pyplot as plt

#import module to set some plotting parameters

from matplotlib import rcParams

#Library to work with Regular Expression

import re

#This Function make plot to show directly in browser

%matplotlib inline

#Setting parameters for the plot

rcParams['figure.figsize'] = 10,8
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



print(df_train.info())

print(df_test.info())
df_train.describe()
df_train.head()
df_train["Name"].head(3)
#GettingLooking of Prefix of all passenger

df_train["Title"] = df_train.Name.apply(lambda x: re.search('([A-Z][a-z]+)\.',x).group(1))



plt.figure(figsize=(12,5))

sns.countplot(x='Title', data=df_train)

plt.xlabel("Title", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.title("Title Name Count", fontsize=20)

plt.xticks(rotation=45)

plt.show()
#Doing the same on df_test with regular expressions

df_test["Title"] = df_test.Name.apply(lambda x: re.search('([A-Z][a-z]+)\.',x).group(1))
#Now, I will identify the social status of each title

Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss":       "Miss",

        "Mr":         "Mr",

        "Master":     "Master"

                }



# we map each title to correct category

df_train['Title'] = df_train.Title.map(Title_Dictionary)

df_test['Title'] = df_test.Title.map(Title_Dictionary)
print("Chances to survive based on titles: ")

print(df_train.groupby("Title")["Survived"].mean())
plt.figure(figsize=(12,5))



sns.countplot(x="Title", data=df_train, hue="Survived")

plt.xlabel("Titles", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.title("Title Grouped Count", fontsize=20)

plt.xticks(rotation=45)

plt.show()
age_high_zero_died = df_train[(df_train["Age"]> 0)& (df_train["Survived"]==0)]

age_high_zero_surv = df_train[(df_train["Age"]> 0)& (df_train["Survived"]==1)]



plt.figure(figsize=(10,5))

sns.distplot(age_high_zero_surv["Age"], bins=24, color='g', kde=False)

sns.distplot(age_high_zero_died["Age"], bins=24, color='r', kde=False)

plt.title("Distribution and density by Age", fontsize=20)

plt.xlabel("Age", fontsize=15)

plt.ylabel("Distribution Died and Survived", fontsize=15)

plt.show()
age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]

print(age_group.median())
# printing the total of nulls in Age Feature

df_train["Age"].isnull().sum()
# using the groupby to transform this variables

df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')
df_train["Age"].isnull().sum()
# Let's see the results of imputation



plt.figure(figsize=(12,5))



sns.distplot(df_train["Age"], bins=24, kde=False)

plt.xlabel("Age")

plt.ylabel("No. of passengers")

plt.title("Distribution and density of Age:")

plt.show()
plt.figure(figsize=(12,5))

g = sns.FacetGrid(df_train, col="Survived", size=5)

g = g.map(sns.distplot, "Age", kde=False)

plt.show()
# Using categorical Variable for different Age's

interval = (0, 5, 12, 18, 25, 35, 60, 120)



cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult','Senior']



df_train["Age_cat"] = pd.cut(df_train.Age, interval,labels=cats)



df_train["Age_cat"].head()
#Do the same to test dataset

interval = (0, 5, 12, 18, 25, 35, 60, 120)



cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult','Senior']



df_test["Age_cat"] = pd.cut(df_test.Age, interval,labels=cats)



df_test["Age_cat"].head()
# Describe of categorical Age

# Using pd.crosstab to understand the Survived rate by Age category's



print(pd.crosstab(df_train.Age_cat, df_train.Survived))



# Setting the figure Size

plt.figure(figsize=(12, 10))



# plotting the result

plt.subplot(2,1,1)

sns.countplot("Age_cat", data=df_train, hue="Survived", palette="hls")

plt.xlabel("Age Category's", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("Age Distribution", fontsize=20)



plt.subplot(2,1,2)

sns.swarmplot(x='Age_cat', y="Fare", data=df_train, hue="Survived", palette="hls")

plt.xlabel("Age Category's", fontsize=18)

plt.ylabel("Fare Distribution", fontsize=18)

plt.title("Fare Distribution by Age Category's", fontsize=20)

plt.subplots_adjust(hspace=0.5, top=0.9)

plt.show()
# Setting the figure size

plt.figure(figsize=(12,5))



# Understanding the Fare Distribution

sns.distplot(df_train[df_train.Survived == 0]["Fare"],bins=50, color='r', kde=False)

sns.distplot(df_train[df_train.Survived == 1]["Fare"],bins=50, color='g', kde=False)

plt.title("Fare Distribution by Survived", fontsize=20)

plt.xlabel("Fare", fontsize=18)

plt.ylabel("Density", fontsize=18)

plt.show()
df_train.Fare = df_train.Fare.fillna(-0.5)



#intervals to categorize

quant = (-1, 0, 8, 15, 31, 600)



#Labels without input values

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']



#doing the cut in fare and puting in a new column

df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)



#Description of transformation

print(pd.crosstab(df_train.Fare_cat, df_train.Survived))



plt.figure(figsize=(12,5))



#Plotting the new feature

sns.countplot(x="Fare_cat", hue="Survived", data=df_train, palette="hls")

plt.title("Count of survived x Fare expending",fontsize=20)

plt.xlabel("Fare Cat",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.show()
# Replicate the same to df_test

df_test.Fare = df_test.Fare.fillna(-0.5)



quant = (-1, 0, 8, 15, 31, 1000)

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']



df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)
#Now lets drop the variable Fare, Age and ticket that is irrelevant now

del df_train["Name"]

del df_train["Fare"]

del df_train["Ticket"]

del df_train["Age"]

del df_train["Cabin"]

#same in df_test

del df_test["Fare"]

del df_test["Ticket"]

del df_test["Age"]

del df_test["Cabin"]

del df_test["Name"]
df_train.head()
df_test.head()
# Let see how many people die or survived

print("Total of Survived or not: ")

print(df_train.groupby("Survived")["PassengerId"].count())



plt.figure(figsize=(12,5))

sns.countplot(x="Survived", data=df_train, palette="hls")

plt.title('Total Distribuition by survived or not', fontsize=22)

plt.xlabel('Target Distribuition', fontsize=18)

plt.ylabel('Count', fontsize=18)



plt.show()
print(pd.crosstab(df_train.Survived, df_train.Sex))

plt.figure(figsize=(12, 5))

sns.countplot(x="Sex", data=df_train, hue="Survived", palette="hls")

plt.title('Sex Distribuition by survived or not', fontsize=20)

plt.xlabel('Sex Distribuition',fontsize=17)

plt.ylabel('Count', fontsize=17)



plt.show()
print(pd.crosstab(df_train.Pclass, df_train.Embarked))

plt.figure(figsize=(12,5))

sns.countplot(x="Embarked", data=df_train, hue="Pclass", palette="hls")

plt.title('Embarked x Pclass Count', fontsize=20)

plt.xlabel('Embarked with Pclass', fontsize=17)

plt.ylabel("Count", fontsize=17)

plt.show()
#lets input the NA's with highest frequency

df_train["Embarked"] = df_train["Embarked"].fillna('S')
print(pd.crosstab(df_train.Survived, df_train.Embarked))



plt.figure(figsize=(12,5))



sns.countplot(x="Embarked", data=df_train, hue="Survived",palette="hls")

plt.title('Class Distribuition by survived or not',fontsize=20)

plt.xlabel('Embarked',fontsize=17)

plt.ylabel('Count', fontsize=17)



plt.show()
# Exploring Survivors vs Pclass

print(pd.crosstab(df_train.Survived, df_train.Pclass))



plt.figure(figsize=(12,5))



sns.countplot(x="Pclass", data=df_train, hue="Survived",palette="hls")

plt.xlabel('PClass',fontsize=17)

plt.ylabel('Count', fontsize=17)

plt.title('Class Distribuition by Survived or not', fontsize=20)



plt.show()
g = sns.factorplot(x="SibSp",y="Survived",data=df_train,

                   kind="bar", height = 5, aspect= 1.6, palette = "hls")

g.set_ylabels("Probability(Survive)", fontsize=15)

g.set_xlabels("SibSp Number", fontsize=15)



plt.show()
# Explore Parch feature vs Survived

g = sns.factorplot(x="Parch", y="Survived", data=df_train, kind="bar", size=6, palette="hls")

g = g.set_ylabels("Survival Probability")
#Create a new column and sum the Parch + SibSp + 1 that refers the people self

df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1



df_test["FSize"] = df_test["Parch"] + df_test["SibSp"] + 1
print(pd.crosstab(df_train.FSize, df_train.Survived))

sns.factorplot(x="FSize",y="Survived", data=df_train, kind="bar",size=6, aspect=1.6)

plt.show()
del df_train["SibSp"]

del df_train["Parch"]



del df_test["SibSp"]

del df_test["Parch"]
df_train.head()
df_test.head()
df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                          prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)



df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                         prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)
df_train.head()
plt.figure(figsize=(15,12))

plt.title('Correlation of Features for train set')

sns.heatmap(df_train.astype(float).corr(), vmax=1.0, annot=True)

plt.show()
df_train.shape
train = df_train.drop(["Survived","PassengerId"],axis=1)

train_ = df_train["Survived"]



test_ = df_test.drop(["PassengerId"],axis=1)



X_train = train.values

y_train = train_.values



X_test = test_.values

X_test = X_test.astype(np.float64, copy=False)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras

from keras.optimizers import SGD

import graphviz
# Creating the model

model = Sequential()

# Inputing the first layer input dimensions

model.add(Dense(18, activation='relu', input_dim=20,

               kernel_initializer='uniform'))

# The argument being passed to each Dense layer(18) is the number of hidden units of the layer

# A hidden unit is a dimensiion in the representatiion space of fhe layer

# Stacks of Dense layers with relu activations can solve a wide range of problems

#(including sentiment classification), and you'll likely use tehm frequently



# Addding an Dropout Layer toprevine from overfitting

model.add(Dropout(0.50))



#Adding second hidden layer

model.add(Dense(60, kernel_initializer='uniform',activation='relu'))



# Adding another Dropout layer

model.add(Dropout(0.50))



#Adding the output layer that is binary [0,1]

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# visualize the model

model.summary()



#Creating an stochastic Gradient Descent

sgd = SGD(lr=0.01, momentum=.09)



# Compiling our model

model.compile(optimizer=sgd, loss='binary_crossentropy',

             metrics = ['accuracy'])



#optimizers list

#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']



# fitting the ANN to the training set

model.fit(X_train, y_train, batch_size=60,epochs=30,verbose=2)
y_preds = model.predict(X_test)



submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')

submission['Survived'] = y_preds.astype(int)

submission.to_csv('output.csv')
scores = model.evaluate(X_train, y_train, batch_size=30)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
history = model.fit(X_train, y_train, validation_split=0.20, epochs=180, batch_size=10, verbose=0)

print(history.history.keys())
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_preds = model.predict(X_test)
# Trying to implementing the TensorBoard to evaluate the model



callbacks = [

    keras.callbacks.TensorBoard(log_dir='my_log_dir',

                                histogram_freq=1,

                                embeddings_freq=1,

                               )

]
# Importing the auxiliar and preprocessing librarys

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.metrics import accuracy_score

#Models 

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.linear_model import RidgeClassifier, SGDClassifier,LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
clfs = []

seed = 3

clfs.append(("LogReg", Pipeline([("Scaler", StandardScaler()),("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier", Pipeline([("Scaler", StandardScaler()),("XGB", XGBClassifier())])))



clfs.append(("KNN", Pipeline([("Scaler", StandardScaler()),("KNN", KNeighborsClassifier())])))



clfs.append(("DecisionTreeClassifier", Pipeline([("Scaler", StandardScaler()),("DecisionTrees", DecisionTreeClassifier())])))



clfs.append(("RandomForestClassifier", Pipeline([("Scaler", StandardScaler()),("RandomForest", RandomForestClassifier())])))



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))



clfs.append(("BaggingRidgeClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("BaggingClassifier", BaggingClassifier())])))



clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreesClassifier())])))



#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'accuracy'

n_folds = 10



results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), 

cv_results.std())

    print(msg)

    

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,6))

fig.suptitle('Classifier Algorithm Comparison', fontsize=22)

ax = fig.add_subplot(111)

sns.boxplot(x=names, y=results)

ax.set_xticklabels(names)

ax.set_xlabel("Algorithmn", fontsize=20)

ax.set_ylabel("Accuracy of Models", fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

plt.show()