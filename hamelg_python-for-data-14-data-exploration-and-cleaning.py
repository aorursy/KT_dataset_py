# Load in some packages

%matplotlib inline





import numpy as np

import pandas as pd

import os
titanic_train = pd.read_csv("../input/train.csv")      # Read the data
titanic_train.shape      # Check dimensions
titanic_train.dtypes
titanic_train.head(5)  # Check the first 5 rows
titanic_train.describe()
categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index

print(categorical)



titanic_train[categorical].describe()
# VARIABLE DESCRIPTIONS:

# survival        Survival

#                 (0 = No; 1 = Yes)

# pclass          Passenger Class

#                 (1 = 1st; 2 = 2nd; 3 = 3rd)

# name            Name

# sex             Sex

# age             Age

# sibsp           Number of Siblings/Spouses Aboard

# parch           Number of Parents/Children Aboard

# ticket          Ticket Number

# fare            Passenger Fare

# cabin           Cabin

# embarked        Port of Embarkation

#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
del titanic_train["PassengerId"]     # Remove PassengerId
sorted(titanic_train["Name"])[0:15]   # Check the first 15 sorted names
titanic_train["Name"].describe()
titanic_train["Ticket"][0:15]       # Check the first 15 tickets
titanic_train["Ticket"].describe()
del titanic_train["Ticket"]        # Remove Ticket
titanic_train["Cabin"][0:15]       # Check the first 15 tickets
titanic_train["Cabin"].describe()  # Check number of unique cabins
new_survived = pd.Categorical(titanic_train["Survived"])

new_survived = new_survived.rename_categories(["Died","Survived"])              



new_survived.describe()
new_Pclass = pd.Categorical(titanic_train["Pclass"],

                           ordered=True)



new_Pclass = new_Pclass.rename_categories(["Class1","Class2","Class3"])     



new_Pclass.describe()
titanic_train["Pclass"] = new_Pclass
titanic_train["Cabin"].unique()   # Check unique cabins
char_cabin = titanic_train["Cabin"].astype(str) # Convert data to str



new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter



new_Cabin = pd.Categorical(new_Cabin)



new_Cabin .describe()
titanic_train["Cabin"] = new_Cabin
dummy_vector = pd.Series([1,None,3,None,7,8])



dummy_vector.isnull()
titanic_train["Age"].describe()
missing = np.where(titanic_train["Age"].isnull() == True)

missing
len(missing[0])
titanic_train.hist(column='Age',    # Column to plot

                   figsize=(9,6),   # Plot size

                   bins=20)         # Number of histogram bins
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check

                       28,                       # Value if check is true

                       titanic_train["Age"])     # Value if check is false



titanic_train["Age"] = new_age_var 



titanic_train["Age"].describe()
titanic_train.hist(column='Age',    # Column to plot

                   figsize=(9,6),   # Plot size

                   bins=20)         # Number of histogram bins
titanic_train["Fare"].plot(kind="box",

                           figsize=(9,9))
index = np.where(titanic_train["Fare"] == max(titanic_train["Fare"]) )



titanic_train.loc[index]
titanic_train["Family"] = titanic_train["SibSp"] + titanic_train["Parch"]
most_family = np.where(titanic_train["Family"] == max(titanic_train["Family"]))



titanic_train.loc[most_family]