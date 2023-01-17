# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)
# reading csv train data into dataframe

train_df = pd.read_csv("../input/train.csv")

# preview train data

train_df.head()
train_df.tail()
#reading csv test data into dataframe

test_df = pd.read_csv("../input/test.csv")

# preview test data

test_df.head()
test_df.tail()
# printing the total sample in train data

print("#of sample in the train data is {}".format(train_df.shape[0]))
# printing the total sample in the test data

print("#of sample in the test data is {}".format(test_df.shape[0]))
# checking missing values in the train data

train_df.isnull().sum()
import missingno as msno
msno.matrix(train_df)
msno.matrix(test_df)
msno.heatmap(train_df)
msno.heatmap(test_df)
msno.dendrogram(train_df)
##percent of missing values in Train data

# ~20% of the Age entries are missing for passengers

# look at what age variable in general
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='blue', alpha=0.6)

train_df["Age"].plot(kind='density', color='brown')

ax.set(xlabel='Age')

plt.xlim(-10,90)

plt.show()
#mean age

print("the mean of age is %.2f" %(train_df['Age'].mean(skipna = True)))
# median of age

print('the median of age is %.2f' %(train_df["Age"].median(skipna = True)))
print('percent of missing "Cabin" record is %.2f%%' %((train_df["Cabin"].isnull().sum()/train_df.shape[0])*100))
print("percent of missing 'Embarked' record is %.2f%%" %((train_df["Embarked"].isnull().sum()/train_df.shape[0])*100))
#Dividing the class of passengers in Train Data

print("Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):")

print(train_df['Embarked'].value_counts())

sns.countplot(x='Embarked', data =train_df, palette= 'Set2')

plt.show()
#Finding the common boarding among the three from train data

print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())
# In a given row if Age is the missing value. Then i will impute median of Age

# If Embarked is the missing value in the row then i will impute as S

# i will ignore Cabnit bec. there is too much missing values in cabnit

train_data = train_df.copy()

train_data['Age'].fillna(train_df["Age"].median(skipna=True), inplace= True)

train_data["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(), inplace=True)

train_data.drop("Cabin", axis=1, inplace=True)
# Checking the missing values in Adjusted data

train_data.isnull().sum()
# preview adj train data

train_data.head()
plt.figure(figsize=(15,5))

ax = train_df['Age'].hist(bins=15, density=True, stacked = True, color = "green", alpha=0.6)

train_df["Age"].plot(kind='density', color = "green")

ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='yellow', alpha=0.5)

train_data['Age'].plot(kind='density', color = 'yellow')

ax.legend(['Raw Age', 'Adjusted Age'])

ax.set(xlabel='Age')

plt.xlim(-10,90)

plt.show()
## Create categorical variable for traveling alone

train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)
#we will also create a categorical variable for passenger-("pclass") Gender-"sex" and port Embarked = "Embarked"



# creating categorical variable and dropping some variables

training=pd.get_dummies(train_data, columns=["Pclass", "Embarked", "Sex"])

training.drop("Sex_female", axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop("Ticket", axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)



final_train = training

final_train.head()
test_df.isnull().sum()
test_data = test_df.copy()

test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

test_data['Fare'].fillna(train_df["Fare"].median(skipna=True), inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)
test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)



test_data.drop("SibSp", axis=1, inplace=True)

test_data.drop("Parch", axis=1, inplace=True)
testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])

testing.drop('Sex_female', axis=1, inplace=True)

testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)

final_test = testing

final_test.head()
# Explanatering & analysing The Data
#Exploaring the Age Data

plt.figure(figsize=(16,9))

ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="lightgreen", shade=True)

sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="orange", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

ax.set(xlabel='Age')

plt.xlim(-10,90)

plt.show()
# so The passengers evidently made an attempt to save children by giving them a place on the life rafts.

plt.figure(figsize=(23,12))

avg_survival_byage = final_train[["Age", "Survived"]].groupby(["Age"], as_index=False).mean()

g = sns.barplot(x="Age", y= "Survived", data=avg_survival_byage, color = "Black")

plt.show()
# From the Bar plot i'll considering the survival rate of passengers under 17, 

# I'll also include another categorical variable in my dataset: Minor

final_train["IsMinor"]=np.where(final_train["Age"]<=17, 1, 0)

#we'll look at test data (it,s my assumption for considering miner = under 17)

final_test["IsMinor"]=np.where(final_test["Age"]<=17, 1, 0)
plt.figure(figsize=(15,8))

ax = sns.kdeplot(final_train["Fare"][final_train.Survived ==1], color= "darkgreen", shade = True)

sns.kdeplot(final_train["Fare"][final_train.Survived ==0], color = "lightcoral", shade =True)

plt.legend(["Survived", "Died"])

plt.title('Density Plot of Fare for Surviving Population And Deceased Population')

ax.set(xlabel="Fare")

plt.xlim(-25,250)

plt.show()
# Exploration of Passengers Class

sns.barplot('Pclass', 'Survived', data= train_df, color="yellow")

plt.show()
##Exploration of Embarked part



sns.barplot('Embarked', 'Survived', data=train_df, color="green")

plt.show()
##Exploration with alone and withFamily

sns.barplot('TravelAlone', 'Survived', data=final_train, color="yellow")

plt.show()
##Explonaration of Gender Variable

sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")

plt.show()