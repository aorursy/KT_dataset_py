#    Name: Complete beginner's first look at Titanic's data

#    Author: Mariana Makiuchi



import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt



train_path = "/kaggle/input/titanic/train.csv"
# Read the training data

train_data = pd.read_csv(train_path)



# Define labels of interest

labels = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Replace the string labels "Sex" and "Embarked" for integer labels

# We do that to plot the histograms for these labels

# using the pandas "hist" function later

train_data["Sex"].replace("female", 0, inplace=True)

train_data["Sex"].replace("male", 1, inplace=True)



train_data["Embarked"].replace("C", 0, inplace=True)

train_data["Embarked"].replace("Q", 1, inplace=True)

train_data["Embarked"].replace("S", 2, inplace=True)



# Separate the data according to whether the passenger survived or not

not_survived = train_data.loc[train_data["Survived"] == 0] 

survived = train_data.loc[train_data["Survived"] == 1]



# Showing basic survival information

print("Number of passengers that survived: ", len(survived))

print("Number of passengers that did not survive: ", len(not_survived))

print("Survival rate: " + str(round(len(survived)*100/len(train_data), 2)) + "%")
# Plotting histograms for the group of passegers that survived

fig1 = plt.figure(figsize = (15,15))

ax1 = fig1.gca()

hist = survived[labels].hist(bins=int(math.sqrt(len(survived))), ax=ax1)
# Plotting histograms for the group of passegers that did not survive

fig2 = plt.figure(figsize = (15,15))

ax2 = fig2.gca()

hist = not_survived[labels].hist(bins=int(math.sqrt(len(not_survived))), ax=ax2, color="red")
# Show statistics about women survival rates

print("Number of women that survived: ", len(survived.loc[survived["Sex"] == 0]))

print("Number of women that did not survive: ", len(not_survived.loc[not_survived["Sex"] == 0]))

print("Women survival rate: "+ str(round(len(survived.loc[survived["Sex"] == 0])*100/len(train_data.loc[train_data["Sex"] == 0]),2)) + "%\n")



# Show statistics of children survival rates (i.e., under 18 years old)

print("Number of children that survived: ", len(survived.loc[survived["Age"] < 18]))

print("Number of children that did not survive: ", len(not_survived.loc[not_survived["Age"] < 18]))

print("Children survival rate: " + str(round(len(survived.loc[survived["Age"] < 18])*100/len(train_data.loc[train_data["Age"] < 18]),2)) + "%\n")



# Show statistics of babies survival rates (i.e., under 4 years old)

print("Number of babies that survived: ", len(survived.loc[survived["Age"] < 4]))

print("Number of babies that did not survive: ", len(not_survived.loc[not_survived["Age"] < 4]))

print("Babies survival rate: " + str(round(len(survived.loc[survived["Age"] < 4])*100/len(train_data.loc[train_data["Age"] < 4]),2)) + "%")
print("Number of elderly people that survived: ", len(survived.loc[survived["Age"] >= 65]))

print("Number of elderly people that did not survive: ", len(not_survived.loc[not_survived["Age"] >= 65]))

print("Elderly people survival rate: " + str(round(len(survived.loc[survived["Age"] >= 65])*100/len(train_data.loc[train_data["Age"] >= 65]),2)) + "%")
# First class

print("Number of 1st class passengers that survived: ", len(survived.loc[survived["Pclass"] == 1]))

print("Number of 1st class passengers that did not survive: ", len(not_survived.loc[not_survived["Pclass"] == 1]))

print("1st class passengers survival rate: " + str(round(len(survived.loc[survived["Pclass"] == 1])*100/len(train_data.loc[train_data["Pclass"] == 1]),2)) + "%\n")



# Second class

print("Number of 2nd class passengers that survived: ", len(survived.loc[survived["Pclass"] == 2]))

print("Number of 2nd class passengers that did not survive: ", len(not_survived.loc[not_survived["Pclass"] == 2]))

print("1st class passengers survival rate: " + str(round(len(survived.loc[survived["Pclass"] == 2])*100/len(train_data.loc[train_data["Pclass"] == 2]),2)) + "%\n")



# Third class

print("Number of 3rd class passengers that survived: ", len(survived.loc[survived["Pclass"] == 3]))

print("Number of 3rd class passengers that did not survive: ", len(not_survived.loc[not_survived["Pclass"] == 3]))

print("1st class passengers survival rate: " + str(round(len(survived.loc[survived["Pclass"] == 3])*100/len(train_data.loc[train_data["Pclass"] == 3]),2)) + "%\n")
# Family size as the sum of all family members

survived["FamilySize"]=survived["SibSp"]+survived["Parch"]

not_survived["FamilySize"]=not_survived["SibSp"]+not_survived["Parch"]



fig1 = plt.figure(figsize = (5,5))

ax1 = fig1.gca()

hist = survived["FamilySize"].hist(bins=int(math.sqrt(len(survived))), ax=ax1)



fig2 = plt.figure(figsize = (5,5))

ax2 = fig2.gca()

hist = not_survived["FamilySize"].hist(bins=int(math.sqrt(len(not_survived))), ax=ax2, color="red")