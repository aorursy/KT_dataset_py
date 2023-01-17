# Import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline
# Collecting data

# The train data set will be used to train our model 
# We will test our model with the test data set
#train_data = pd.read_csv("train.csv")
#test_data = pd.read_csv("test.csv")

# Kaggle

train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head()
test_data.head()
print(train_data.shape,test_data.shape)
# Survived/Deceased v Sex and Pclass

f, axes = plt.subplots(nrows=1,ncols=2, figsize=(8, 4))
sns.countplot(x="Survived",data=train_data,hue="Sex",ax=axes[0])
sns.countplot(x="Survived",data=train_data,hue="Pclass",ax=axes[1])
# Siblings or spouses aboard

plt.figure(figsize=(12,6))
sns.countplot(x="Survived",data=train_data,hue="SibSp")
# Parents or children aboard

plt.figure(figsize=(12,6))
sns.countplot(x="Survived",data=train_data,hue="Parch")
# Age 

plt.figure(figsize=(12,6))
sns.distplot(train_data["Age"],bins=60)
train_data.describe()
train_data.head()
# Missing data

print(train_data.isnull().sum())
print("\n")
print(test_data.isnull().sum())
# Sex
# We need to transform words (male / female) into numbers so that computer understands
# We could have used the function pandas.get_dummies

train_data.loc[train_data["Sex"] == "male","Sex"] = 1
train_data.loc[train_data["Sex"] == "female","Sex"] = 0

# Test data

test_data.loc[train_data["Sex"] == "male","Sex"] = 1
test_data.loc[train_data["Sex"] == "female","Sex"] = 0
# Embarked
# We are missing 2 values, we will fill them with "S" because it is the most common
# As we did before, we will turn letters into numbers so that the model can work with them

# Train data

train_data["Embarked"] = train_data["Embarked"].fillna("S")
train_data.loc[train_data["Embarked"] == "S","Embarked"] = 0
train_data.loc[train_data["Embarked"] == "C","Embarked"] = 1
train_data.loc[train_data["Embarked"] == "Q","Embarked"] = 2

# Test data

test_data.loc[train_data["Embarked"] == "S","Embarked"] = 0
test_data.loc[train_data["Embarked"] == "C","Embarked"] = 1
test_data.loc[train_data["Embarked"] == "Q","Embarked"] = 2
titles = []
for i in train_data["Name"]:
    titles.append(i.split(",")[1].split(".")[0].strip())
    
print(set(titles))
Title_dict = {
    "Capt": 3,
    "Col": 3,
    "Major": 3,
    "Jonkheer": 3,
    "Don": 3,
    "Sir" : 3,
    "Dr": 3,
    "Rev": 3,
    "the Countess":3,
    "Mme": 3,
    "Mlle": 3,
    "Ms": 2,
    "Mr" : 0,
    "Mrs" : 2,
    "Miss" : 1,
    "Master" : 3,
    "Lady" : 3
}

# 3 = Highest rank royalty/officer
# 2 = Married women
# 1 = Unmarried women
# 0 = Mr

print(Title_dict)

# The titles need to be numbered so that the model can process them
# Add new column to data set with titles

def get_titles(data_set):
    # we extract the title from each name
    data_set['Title'] = data_set['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    data_set['Title'] = data_set.Title.map(Title_dict)
    return data_set
# Train data
train_data = get_titles(train_data)

# Test data
test_data = get_titles(test_data)
test_data.isnull().sum()
# There is one missing title in our test data. We need to fill in the missing Title value.

print("Number of titles in the test data:")
print(test_data["Title"].value_counts())
print("")
name = str(test_data[test_data["Title"].isna()]["Name"])
print(f"The missing title corresponds to: {name}")
test_data[test_data["Title"].isna()]
# How should we fill the missing Title?

test_data["Title"].value_counts()
# We will fill the missing Title value with the no. 1, since the missing passenger is a female travelling alone (Miss.)

test_data["Title"] = test_data["Title"].fillna(1)
if train_data["Title"].isnull().sum() == 0 and test_data["Title"].isnull().sum() == 0:
    
    print("The Titles in the train and test data sets are now complete!")
# Age: Fill the missing
# We will use the mean age to fill in the missing age values

# Train data
# train_data["Age"] = train_data["Age"].fillna(round(train_data["Age"].mean()))

# Test data
# test_data["Age"] = test_data["Age"].fillna(round(train_data["Age"].mean()))

# Let's see what features correlate best with age

train_data.corr()["Age"].sort_values()
plt.figure(figsize=(12,6))
sns.boxplot(x="Title",y="Age",data=train_data)
plt.figure(figsize=(12,6))
sns.boxplot(x="Pclass",y="Age",data=train_data)
train_data.groupby("Title").mean()["Age"]
# Fill the age according to the Title

# Train data
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"),inplace=True)

# Test data (we will use the mean of the train data as well)
test_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"),inplace=True)
if train_data["Age"].isnull().sum() == 0 and test_data["Age"].isnull().sum() == 0:
    
    print("The Ages in the train and test data sets are now complete!")
train_data["Cabin"].value_counts()
# Cabin numbers that we are missing
train_data["Cabin"].isnull().sum(),test_data["Cabin"].isnull().sum()
# We will start with the train data

# We can see that most of the passengers from first class have a cabin number, whereas the third class ones don´t.

train_data_cabin_null = train_data[train_data["Cabin"].isna()][["Pclass","Cabin","Fare"]]
train_data_cabin_not_null = train_data[train_data["Cabin"].notna()][["Pclass","Cabin","Fare"]]

train_data_cabin_null

train_data_cabin_null["Pclass"].value_counts()
x = ["Third class","Second class","First class"]
y = list(train_data_cabin_null["Pclass"].value_counts())
plt.title("Missing Cabin numbers for each Passenger class\nTrain data")
plt.xlabel("Passenger Class")
plt.ylabel("Number of missing cabin names")

plt.bar(x,y)
#train_data[train_data["Pclass"]==1][["Pclass","Cabin","Fare"]].head(10)
# Dataframe with passengers that have a cabin number

train_data_cabin_not_null.head(10)
def get_number_of_cabins(string):
    number_of_cabins = 1
    for i in string:
        if i == " ":
            number_of_cabins += 1
            
    return number_of_cabins

def first_letter(string):
    return string[0]
# I´m only interested in the first letter to know the deck
# Some fares correspond to more than one cabin
# Find out fare per cabin 

train_data_cabin_not_null["No_cabins"] = 1 # Fill with 1 at the begining
train_data_cabin_not_null["Deck"] = "A"

# Deck letters

for row in train_data_cabin_not_null.index:
    train_data_cabin_not_null["No_cabins"][row] = get_number_of_cabins(train_data_cabin_not_null["Cabin"][row])
    train_data_cabin_not_null["Deck"][row] = first_letter(train_data_cabin_not_null["Cabin"][row])

# Mean price per cabin
    
train_data_cabin_not_null["Price_per_cabin"] = train_data_cabin_not_null["Fare"] / train_data_cabin_not_null["No_cabins"]

train_data_cabin_not_null
plt.figure(figsize=(14,8))

sns.swarmplot(x="Deck",y="Price_per_cabin",data=train_data_cabin_not_null,size=6)
train_data_cabin_not_null["Deck"].unique()
for pclass in [1,2,3]:
    decks = train_data_cabin_not_null[train_data_cabin_not_null["Pclass"]==pclass]["Deck"].unique()
    print(f"Passenger class: {pclass}\tDecks: {decks}")
for deck in ["A","B","C","D","E","F","G","T"]:
    mean = train_data_cabin_not_null[train_data_cabin_not_null["Deck"] == deck]["Price_per_cabin"].mean()
    print(f"Mean price per cabin for in Deck: {deck} is {round(mean,3)}")
# Dictionaries

dict_pclass_deck = {
    1:["A","B","C","D","E","T"],
    2:["D","E","F"],
    3:["E","F","G"]
}

dict_fare_deck = {
    "A": 39.624,
    "B": 86.579,
    "C": 82.483,
    "D": 53.285,
    "E": 46.027,
    "F": 16.954,
    "G": 13.581,
    "T": 35.5
}
def get_deck(dict_fare,dict_pclass, fare, pclass):
    listOfItems = dict_fare.items()
    for item  in listOfItems:
        if item[1] == fare:
            deck = item[0]
            return deck

    # We haven´t found the exact value
    difference = list()
    difference_decks = list()
    for item  in listOfItems:
        #print(item[0],abs(valueToFind - item[1]))
        difference.append(abs(fare - item[1]))
        difference_decks.append(item[0])
        
    # Find index of minimum and correspond it to list of decks
    
    minimum_index = difference.index(min(difference))
    possible_deck = difference_decks[minimum_index]
    
    # We have the most probable deck based on the fare
    # Make sure this is possible regarding passenger class
    
    if possible_deck in dict_pclass.get(pclass):
        deck = possible_deck
        
        return deck
    
    else:
        #print(f"Suggest deck: {possible_deck}. But there is no {possible_deck} deck in pclass {pclass}")
        # Two options:
            # 1. Fare to high for class --> Best deck in class
            # 2. Class to high for fare --> Worst deck in class
            
        if fare > dict_fare.get(possible_deck): # Opt. 1
            
            return dict_pclass.get(pclass)[0] # Best deck of its class
        
        else:
            
            return dict_pclass.get(pclass)[-1] # Worst deck of its class
        

train_data["Deck"] = "A"  # Fill with "A" provisionally

# Deck letters

train_data["Cabin"] = train_data["Cabin"].fillna("Z") # Fill missing with "Z" provisionally

# To fill column Deck:
# If cabin != "Z" appply function of getting first letter
# If cabin == "Z" apply get_deck function

for row in train_data.index:
    if train_data["Cabin"][row] != "Z":
        train_data["Deck"][row] = first_letter(train_data["Cabin"][row])
    else:
        fare = train_data["Fare"][row]
        pclass = train_data["Pclass"][row]
        train_data["Deck"][row] = get_deck(dict_fare_deck,dict_pclass_deck,fare,pclass)

train_data.head()
# Turn deck into numeric values

dict_deck_number = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "T": 8
}

def get_deck_no(data_set):
    data_set['Deck_no'] = data_set.Deck.map(dict_deck_number)
    return data_set
train_data = get_deck_no(train_data)
# We now repeat the process with the test data

# 1. Filling the missing values
    
test_data["Deck"] = "A"  # Fill with "A" provisionally

# Deck letters

test_data["Cabin"] = test_data["Cabin"].fillna("Z") # Fill missing with "Z" provisionally

# To fill column Deck:
# If cabin != "Z" appply function of getting first letter
# If cabin == "Z" apply get_deck function

for row in test_data.index:
    if test_data["Cabin"][row] != "Z":
        test_data["Deck"][row] = first_letter(test_data["Cabin"][row])
    else:
        fare = test_data["Fare"][row]
        pclass = test_data["Pclass"][row]
        test_data["Deck"][row] = get_deck(dict_fare_deck,dict_pclass_deck,fare,pclass)

# 2. Turn deck into numeric values

test_data = get_deck_no(test_data)
train_data.isnull().sum()
test_data.isnull().sum()
# We will do the same for the test data as we did before for the train data

test_data.loc[test_data["Sex"] == "male","Sex"] = 1
test_data.loc[test_data["Sex"] == "female","Sex"] = 0
test_data["Embarked"] = test_data["Embarked"].fillna("S")
test_data.loc[test_data["Embarked"] == "S","Embarked"] = 0
test_data.loc[test_data["Embarked"] == "C","Embarked"] = 1
test_data.loc[test_data["Embarked"] == "Q","Embarked"] = 2
#test_data["Age"] = test_data["Age"].fillna(round(test_data["Age"].mean()))
test_data["Fare"] = test_data["Fare"].fillna(round(train_data["Fare"].mean()))
# Title information see bellow

# We filled in the missing fare value with the mean from the train data where we have all values
test_data.head()
print(train_data.isnull().sum())
print("")
print(test_data.isnull().sum())
#train_data.drop("Name",axis=1,inplace=True)
#test_data.drop("Name",axis=1,inplace=True)
# Machine learning libraries

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
titanic_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Title","Embarked","Deck_no"]

# Adding Embarked as feature does not improve our model

X_train = train_data[titanic_features]
y_train = train_data["Survived"]
# Define our model. Specify a number for random_state to ensure same results each run.

titanic_model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

#Fit model
titanic_model.fit(X_train,y_train)

print("Model succesfully created!")
# Making predictions...
# These results should be 1 or 0, alive or dead, we can´t have decimals.

predicted_values = titanic_model.predict(X_train)

predicted_values_rounded = []
for i in predicted_values:
    predicted_values_rounded.append(round(i))
from sklearn.metrics import mean_absolute_error
predicted_values_train = titanic_model.predict(X_train)

MAE_rfc = round(mean_absolute_error(y_train,predicted_values_train),4)
acc_rfc = round(titanic_model.score(X_train,y_train),4)
X_test = test_data[titanic_features]
print("Model Evaluation:")
print(f"Mean Absolute Error: {MAE_rfc}")
print(f"Model Accuracy: {acc_rfc}")
predicted_values_test = titanic_model.predict(X_test)
# Round the predicted values to 0 or 1

predicted_values_test_rounded = []

for i in predicted_values_test:
    predicted_values_test_rounded.append(int(round(i)))
# Save to CSV file

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived":predicted_values_test_rounded
})

submission.to_csv("titanicSub_v.csv",index=False)
submission
print("Model Evaluation:")
print(f"Mean Absolute Error: {MAE_rfc}")
print(f"Model Accuracy: {acc_rfc}")
