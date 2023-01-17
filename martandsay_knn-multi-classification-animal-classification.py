# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_class = pd.read_csv("../input/class.csv")

df_animals = pd.read_csv("../input/zoo.csv")
df_class
# We have 7 different class in class.csv. These classes are the species class of the animals.

df_class.Class_Number.unique()
# Every animal has a class from class.csv

df_animals.head()
# Class wise animal counts.

# We can see, mostly animals belong to the class 1 which is Mammal.

plt.figure(figsize=(10,8));

df_animals.class_type.value_counts().plot(kind="bar");

plt.xlabel('Class Type');

plt.ylabel("Count");

plt.plot();
# Lets plot how many animals are domestic or not

plt.figure(figsize=(10,8));

df_animals.domestic.value_counts().plot(kind="bar");

plt.xlabel('Is Domestic');

plt.ylabel("Count");

plt.plot();
# So we can see mostly animals are not domestic.
pd.crosstab(df_animals.class_type, df_animals.domestic)
# Lets see species wise domestic and non-domestic animals

pd.crosstab(df_animals.class_type, df_animals.domestic).plot(kind="bar", figsize=(10, 8), title="Class wise Domestic & Non-Domestic Count");

plt.plot();
# We can see mammals class has most number of domestic animals, which is a kind of true if you will see around

# you. Mostly domestic animals are mammals like dogs, cats, cows, pigs.
# Lets see how many animals provides us milk

df_animals.milk.value_counts()
# So there are 41 animals in the list which provides us milk. Lets see to which category they belongs
pd.crosstab(df_animals.class_type, df_animals.milk)
# So we can observer here only mammals provides milk, which is really a scientific true. 

# that mean our data exploration is going good till now.. CONGRATS !!!
pd.crosstab(df_animals.class_type, df_animals.milk).plot(kind="bar", title="Class wise Milk providing animals", figsize=(10, 8));
# We can see mammal bar is orange (milk = 1), this shows all the mammals in our list provides milk.

# And no othere class animals gives us milk, Actually our plot really makes sense.

# lets find is it correct.

df_animals[(df_animals.milk==1)].shape[0]
df_animals[df_animals.class_type == 1].shape[0]
# So yes we can see milk animals and mammals have equal numbers and all animals who provide milk belong to 

# mammal category. 
# Lets see how many animals live under water. i.e aquatic

# lets find out all the aquatic animals.

df_animals.aquatic.value_counts() # only 36 aquatic animals are there.

# lets see there class.
df_animals[df_animals.aquatic==1].class_type.value_counts()
# We can see mostly aquatic animals are fish means class 4, but wait we have few animals with 

# class 7, 2, 1, 5, 3

# and still they have fins. 

# Lets find out is it a wrong data or what?

# Lets plot category wise animals having fins

pd.crosstab(df_animals.class_type, df_animals.aquatic).plot(kind="bar", figsize=(10, 8));
# AMAZING NATURE...!!!
# What about venomous?

df_animals.venomous.value_counts()
# In our dataset we have 8 venomous. Lets see their class

pd.crosstab(df_animals.class_type, df_animals.venomous)
# As per our data we have 2 repltiles which are venomous and , 1 fish and other class types

pd.crosstab(df_animals.class_type, df_animals.venomous).plot(kind="bar", figsize=(10, 8))
# From above plot, we know that no mammals and birds are venomous. So thank god you can hug them.

# but better watch your back before mammal kick your... !@#
# So enough EDA i guess, You can also practice few like:

# find out the venomous fish.

# find out venomous reptiles.

# find out venomous animal which gives egg or milk or any kind of data.
df_animals.shape
#X = df_animals.iloc[:, 1: 17].values

X=df_animals.loc[:, ["milk", "backbone", "toothed", "venomous", "domestic", "aquatic"]].values
y = df_animals.iloc[:, 17].values
y.shape
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2) 

# n_neighbors: number of neighbors. Default is 5

# metric="minkowski", p=2: will calculate distance as eucledian distance formula
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred # here you can see our model predict the class of the animal for the test data.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
print("Accuracy of KNN Regression:",accuracy_score(y_test, y_pred))
# So we can see we have an accuracy of 92% which is very good

from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(random_state=0)
clf_log.fit(X_train, y_train)
y_pred_log = clf_log.predict(X_test)
print("Accuracy of Logistic Regression Classifier:",accuracy_score(y_test, y_pred_log))