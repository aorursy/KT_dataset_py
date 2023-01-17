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
from sklearn.neural_network import MLPRegressor

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np

import seaborn as sns

# to render the graphs

import matplotlib.pyplot as plt

# import module to set some ploting parameters

from matplotlib import rcParams

# Library to work with Regular Expressions

import re



# This function makes the plot directly on browser

%matplotlib inline



# Seting a universal figure size 

rcParams['figure.figsize'] = 10,8
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
df_gender.head()
df_test.head()
#Looking how the data is and searching for a re patterns

df_train["Name"].head()
#GettingLooking the prefix of all Passengers

df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))



#defining the figure size of our graphic

plt.figure(figsize=(12,5))



#Plotting the result

sns.countplot(x='Title', data=df_train, palette="hls")

plt.xlabel("Title", fontsize=16) #seting the xtitle and size

plt.ylabel("Count", fontsize=16) # Seting the ytitle and size

plt.title("Title Name Count", fontsize=20) 

plt.xticks(rotation=45)

plt.show()
#Doing the same on df_test with regular expressions

df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
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

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

                   }

    

# we map each title to correct category

df_train['Title'] = df_train.Title.map(Title_Dictionary)

df_test['Title'] = df_test.Title.map(Title_Dictionary)
print("Chances to survive based on titles: ") 

print(df_train.groupby("Title")["Survived"].mean())



# figure size

plt.figure(figsize=(12,5))



#Plotting the count of title by Survived or not category

sns.countplot(x='Title', data=df_train, palette="hls",

              hue="Survived")

plt.xlabel("Titles", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.title("Title Grouped Count", fontsize=20)

plt.xticks(rotation=45)

plt.show()
#I will create a df to look distribuition 

age_high_zero_died = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 0)]

age_high_zero_surv = df_train[(df_train["Age"] > 0) & 

                              (df_train["Survived"] == 1)]



#figure size

plt.figure(figsize=(10,5))



# Ploting the 2 variables that we create and compare the two

sns.distplot(age_high_zero_surv["Age"], bins=24, color='g')

sns.distplot(age_high_zero_died["Age"], bins=24, color='r')

plt.title("Distribuition and density by Age",fontsize=20)

plt.xlabel("Age",fontsize=15)

plt.ylabel("Distribuition Died and Survived",fontsize=15)

plt.show()
#Let's group the median age by sex, pclass and title, to have any idea and maybe input in Age NAN's

age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]



#printing the variabe that we created by median

print(age_group.median())
#inputing the values on Age Na's 

# using the groupby to transform this variables

df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')



# printing the total of nulls in Age Feature

print(df_train["Age"].isnull().sum())
#Let's see the result of the inputation



#seting the figure size

plt.figure(figsize=(12,5))



#ploting again the Age Distribuition after the transformation in our dataset

sns.distplot(df_train["Age"], bins=24)

plt.title("Distribuition and density by Age")

plt.xlabel("Age")

plt.show()

#separate by survivors or not



# figure size

plt.figure(figsize=(12,5))



# using facetgrid that is a great way to get information of our dataset

g = sns.FacetGrid(df_train, col='Survived',size=5)

g = g.map(sns.distplot, "Age")

plt.show()
#df_train.Age = df_train.Age.fillna(-0.5)



#creating the intervals that we need to cut each range of ages

interval = (0, 5, 12, 18, 25, 35, 60, 120) 



#Seting the names that we want use to the categorys

cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']



# Applying the pd.cut and using the parameters that we created 

df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)



# Printing the new Category

df_train["Age_cat"].head()
#Do the same to test dataset 

interval = (0, 5, 12, 18, 25, 35, 60, 120)



#same as the other df train

cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']



# same that we used above in df train

df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=cats)
#Describe of categorical Age



# Using pd.crosstab to understand the Survived rate by Age Category's

print(pd.crosstab(df_train.Age_cat, df_train.Survived))



#Seting the figure size

plt.figure(figsize=(12,10))



#Plotting the result

plt.subplot(2,1,1)

sns.countplot("Age_cat",data=df_train,hue="Survived", palette="hls")

plt.ylabel("Count", fontsize=18)

plt.xlabel("Age Categorys", fontsize=18)

plt.title("Age Distribution ", fontsize=20)



plt.subplot(2,1,2)

sns.swarmplot(x='Age_cat',y="Fare",data=df_train,

              hue="Survived", palette="hls", )

plt.ylabel("Fare Distribution", fontsize=18)

plt.xlabel("Age Categorys", fontsize=18)

plt.title("Fare Distribution by Age Categorys ", fontsize=20)



plt.subplots_adjust(hspace = 0.5, top = 0.9)



plt.show()
Age_fare = ['Pclass', 'Age_cat'] #seting the desired 



cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[Age_fare[0]], df_train[Age_fare[1]], 

            values=df_train['Fare'], aggfunc=['mean']).style.background_gradient(cmap = cm)
# Seting the figure size

plt.figure(figsize=(12,5))



# Understanding the Fare Distribuition 

sns.distplot(df_train[df_train.Survived == 0]["Fare"], 

             bins=50, color='r')

sns.distplot(df_train[df_train.Survived == 1]["Fare"], 

             bins=50, color='g')

plt.title("Fare Distribuition by Survived", fontsize=20)

plt.xlabel("Fare", fontsize=15)

plt.ylabel("Density",fontsize=15)

plt.show()

#Filling the NA's with -0.5

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
df_test.Fare = df_test.Fare.fillna(-0.5)



quant = (-1, 0, 8, 15, 31, 1000)

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']



df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)
#Now lets drop the variable Fare, Age and ticket that is irrelevant now

del df_train["Fare"]

del df_train["Ticket"]

del df_train["Age"]

del df_train["Cabin"]

del df_train["Name"]



#same in df_test

del df_test["Fare"]

del df_test["Ticket"]

del df_test["Age"]

del df_test["Cabin"]

del df_test["Name"]
#Looking the result of transformations

df_train.head()
# Let see how many people die or survived

print("Total of Survived or not: ")

print(df_train.groupby("Survived")["PassengerId"].count())



plt.figure(figsize=(12,5))



sns.countplot(x="Survived", data=df_train,palette="hls")

plt.title('Total Distribuition by survived or not', fontsize=22)

plt.xlabel('Target Distribuition', fontsize=18)

plt.ylabel('Count', fontsize=18)



plt.show()
print(pd.crosstab(df_train.Survived, df_train.Sex))



plt.figure(figsize=(12,5))

sns.countplot(x="Sex", data=df_train, hue="Survived",palette="hls")

plt.title('Sex Distribuition by survived or not', fontsize=20)

plt.xlabel('Sex Distribuition',fontsize=17)

plt.ylabel('Count', fontsize=17)



plt.show()
print(pd.crosstab(df_train.Pclass, df_train.Embarked))



plt.figure(figsize=(12,5))



sns.countplot(x="Embarked", data=df_train, hue="Pclass",palette="hls")

plt.title('Embarked x Pclass Count', fontsize=20)

plt.xlabel('Embarked with PClass',fontsize=17)

plt.ylabel('Count', fontsize=17)



plt.show()
#lets input the NA's with the highest frequency

df_train["Embarked"] = df_train["Embarked"].fillna('S')
# Exploring Survivors vs Embarked

print(pd.crosstab(df_train.Survived, df_train.Embarked))



plt.figure(figsize=(12,5))



sns.countplot(x="Embarked", data=df_train, hue="Survived",palette="hls")

plt.title('Class Distribuition by survived or not',fontsize=20)

plt.xlabel('Embarked',fontsize=17)

plt.ylabel('Count', fontsize=17)



plt.show()
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

g  = sns.factorplot(x="Parch",y="Survived",data=df_train, kind="bar", size = 6,palette = "hls")

g = g.set_ylabels("survival probability")
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
df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                          prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)



df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\

                         prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)
plt.figure(figsize=(15,12))

plt.title('Correlation of Features for Train Set')

sns.heatmap(df_train.astype(float).corr(),vmax=1.0,  annot=True)

plt.show()
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



# Inputing the first layer with input dimensions

model.add(Dense(18, 

                activation='relu',  

                input_dim=20,

                kernel_initializer='uniform'))

#The argument being passed to each Dense layer (18) is the number of hidden units of the layer. 

# A hidden unit is a dimension in the representation space of the layer.



#Stacks of Dense layers with relu activations can solve a wide range of problems

#(including sentiment classification), and you’ll likely use them frequently.



# Adding an Dropout layer to previne from overfitting

model.add(Dropout(0.50))



#adding second hidden layer 

model.add(Dense(25,

                kernel_initializer='uniform',

                activation='relu'))



# Adding another Dropout layer

model.add(Dropout(0.50))



model.add(Dense(20,

                kernel_initializer='uniform',

                activation='relu'))

# Adding another Dropout layer

model.add(Dropout(0.50))



# adding the output layer that is binary [0,1]

model.add(Dense(1,

                kernel_initializer='uniform',

                activation='sigmoid'))

#With such a scalar sigmoid output on a binary classification problem, the loss

#function you should use is binary_crossentropy



#Visualizing the model

model.summary()

#Creating an Stochastic Gradient Descent

sgd = SGD(lr = 0.001, momentum = 0.9)



# Compiling our model

model.compile(optimizer = 'adam', 

                   loss = 'binary_crossentropy', 

                   metrics = ['accuracy'])

#optimizers list

#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']



# Fitting the ANN to the Training set

model.fit(X_train, y_train, 

               batch_size = 60, 

               epochs = 35, verbose=2)
y_preds = model.predict(X_test)



submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv", index_col='PassengerId')

submission['Survived'] = y_preds.astype(int)

submission.to_csv('TitanicKNN.csv')
scores = model.evaluate(X_train, y_train, batch_size=30)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Fit the model

history = model.fit(X_train, y_train, validation_split=0.20, 

                    epochs=180, batch_size=10, verbose=0)



# list all data in history

print(history.history.keys())
# summarizing historical accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()