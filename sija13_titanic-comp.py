# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Let's load our data with pandas

train_data = pd.read_csv("../input/train.csv")

print(type(train_data))
#Let us look what kind of shape we have and what our data says to have an understanding of it

print(train_data.shape)
train_data.head()
#Let's see also what data we are dealing with

train_data.dtypes
#I will resolve to using decision trees to make predictions, but for that I also need prepared data. First

#I want to check if there is missing data and then also reduce the table to work with only to the dimensions that seem actually

#matter.

train_data.isnull().sum()

#Well, there are few missing values...which is not a good thing for us as Age is definitelly one of the dimensions

#that could matter highly on this

#Interpolation on this might not be a good idea, but then that means we would want to drop it for future use
#Let us create a new data frame

df_train = train_data[["Pclass", "Sex", "Fare"]] #from theoretical approach - these 

                                                 #are the main factors that can matter

                                                 #in determening whether someone survived or not

#And we need our labels        

df_label = train_data[["Survived"]]
df_train.head()
df_label.head()
#It would be really good to convert Sex into integers say 0 = male, 1 = female

df_train.dtypes
#We need to check if everything is written in the same fashion "male" and no "Male" or "mALE" and so on.

df_train.Sex.unique()
df_train.loc[df_train["Sex"] == 'male', "Sex"] = 0

df_train.loc[df_train["Sex"] == 'female', "Sex"] = 1
#Alright, everything changed, let's make our training system

print(df_train.dtypes)

df_train.head(10)
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

#We will use a standart sklearn model

clf = RandomForestClassifier(n_estimators=10000, max_depth=4, random_state=666)



#And we want some data for validation later

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, df_label, test_size=0.10, random_state=1337)
#Let's see what goes into our splitted variables for training and testing

print(X_train.head())

print(y_train.head())

print()

print("Amount of records:", len(X_train))
#Alright, now let's train our decision tress based system

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
#And also let's try K-nn

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train) 
neigh.score(X_test, y_test)
#We can easily experiment with the K-nn

neigh_list = [1,3,5,7,9,11,12]

for i in neigh_list:

    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train, y_train) 

    print(i, neigh.score(X_test, y_test))

    

#Accuracy is best on 7 neighbours
#And lets look at SVMs

from sklearn.svm import SVC

svm_clf = SVC(gamma='auto')

svm_clf.fit(X_train, y_train) 
svm_clf.score(X_test, y_test)
array_predictions = clf.predict(X_test)
import numpy as np

y_test_array = np.array(y_test)
print(array_predictions.shape)

print(y_test_array.shape)
y_test_array_new = y_test_array.T[0]

y_test_array_new
print(array_predictions.shape)

print(y_test_array_new.shape)
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')



bins = np.linspace(0, 1, 4)



plt.hist([array_predictions, y_test_array_new], bins, label=['Predicted', 'Real'])

plt.legend(loc='best')

plt.show()



#It predicts that a bit more people died than there should be
#The issue here will be once more with loading and cleaning the data slightly to our liking. Luckily, this also means we can re-use our code from the before.

df_test = pd.read_csv("../input/test.csv")
df_test.head()
#Update our information and quickly glance at our data

df_test.loc[df_test["Sex"] == 'male', "Sex"] = 0

df_test.loc[df_test["Sex"] == 'female', "Sex"] = 1

df_test_pred = df_test[["Pclass", "Sex", "Fare"]]

print(df_test_pred.dtypes)

print()

print("Missing values information:")

print(df_test_pred.isnull().sum())
#oh no there is one missing point for the fare...let's cheat and take an average for it just because there is only 1 missing point and not many

#Let's take a look where the missing point of info is

null_data = df_test_pred[df_test_pred.isnull().any(axis=1)]

null_data

#Row 152, good, let's remember it
#To check if we are right

df_test_pred.iloc[152]

#We are swagaliciously right
#Let's fix it and check

df_test_pred["Fare"].fillna(df_test_pred["Fare"].mean(), inplace=True)

df_test_pred.iloc[152]

#Well...this more or less works
#Now let's use our model for making of quick predictions // Naturally we need to save our output somewhere

predictions_variable = clf.predict(df_test_pred)
#and as we can see all of our predictions got saved into a nice array.

predictions_variable
#Now let us add them to the file...and let's try using pandas for that as well. We are learning :) 

#Do notice that we need to include passengersID as well. So we need preparation of that.

#Even though numbers go from 1 to ..., in the real world it would be better to show such practices where you assume you do not know that and numbers could be anything.

idx = df_test["PassengerId"]
#And let's set our both variables to arrays to work with them in the same format

import numpy as np

idx_array = np.array(idx)

idx_array
#Let's create a dataframe to hold these both variables

final_df = pd.DataFrame({'PassengerId':idx_array, 'Survived':predictions_variable})
#Let's take a look at our prepared data

final_df.head(15)

#Great, now we can save it into a file
#This should do the trick

final_df.to_csv('Titanic_Submission.csv', index=False, header=True)