# important pakages

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np
features_train = pd.read_csv("../input/train.csv")
features_train.shape
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100
features_train.head()
features_train.describe()
features_train["Age"]
features_train["Age"].fillna(features_train["Age"].mean(), inplace = True)
features_train.describe()
features_train["Age"]
import matplotlib

matplotlib.style.use('ggplot')
survived_sex = features_train[features_train["Survived"] == 1]["Sex"].value_counts()

dead_sex = features_train[features_train["Survived"] == 0]["Sex"].value_counts()

df = pd.DataFrame([survived_sex, dead_sex])

df.index = ["Survived", "Dead"]

df.plot(kind = "bar", stacked = True, figsize=(10,8))

plt.ylabel("Number of Passanger")
figure = plt.figure(figsize =(15,8) )

plt.hist([features_train[features_train["Survived"] == 1]["Age"], 

          features_train[features_train["Survived"] == 0]["Age"]],stacked=True,

         color = ['g', 'r'],bins = 30, label = ['Survived', 'Dead'],)

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
figure = plt.figure(figsize =(15,8) )

plt.hist([features_train[features_train["Survived"] == 1]["Fare"], 

          features_train[features_train["Survived"] == 0]["Fare"]],stacked=True,

         color = ['g', 'r'],bins = 30, label = ['Survived', 'Dead'],)

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(features_train[features_train["Survived"] == 1]["Age"], 

           features_train[features_train["Survived"] == 1]["Fare"], c = 'green', s = 40)

ax.scatter(features_train[features_train["Survived"] == 0]["Age"], 

           features_train[features_train["Survived"] == 0]["Fare"], c = 'red', s = 40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('Survived','Dead'), scatterpoints=1, loc='upper right', fontsize=15,)
ax = plt.subplot()

ax.set_ylabel('Average fare')

features_train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
survived_embark = features_train[features_train['Survived']==1]['Embarked'].value_counts()

dead_embark = features_train[features_train['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar', stacked=True, figsize=(15,8))

plt.ylabel('Number of Passangers')
def get_combined_data():

    features_train = pd.read_csv("../input/train.csv")

    features_test = pd.read_csv("../input/test.csv")

    # remove the survived coloumn from training dataset

    targets = features_train.Survived

    features_train.drop('Survived',1,inplace=True)

    # combined the training and testing data set

    combined = features_train.append(features_test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True,axis=1)

    return combined
combined = get_combined_data()
combined.head()
combined
print(combined["Name"][400:500])