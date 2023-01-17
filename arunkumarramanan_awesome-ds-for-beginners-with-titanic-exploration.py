import pandas as pd
# Read from CSV to Pandas DataFrame

df = pd.read_csv("../input/train.csv", header=0)
# First five items

df.head()
# Describe features

df.describe()
# Histograms

df["Age"].hist()
# Unique values

df["Embarked"].unique()
# Selecting data by feature

df["Name"].head()
# Filtering

df[df["Sex"]=="female"].head() # only the female data appear
# Sorting

df.sort_values("Age", ascending=False).head()
# Grouping

sex_group = df.groupby("Survived")

sex_group.mean()
# Selecting row

df.iloc[0, :] # iloc gets rows (or columns) at particular positions in the index (so it only takes integers)
# Selecting specific value

df.iloc[0, 1]
# Selecting by index

df.loc[0] # loc gets rows (or columns) with particular labels from the index
# Rows with at least one NaN value

df[pd.isnull(df).any(axis=1)].head()
# Drop rows with Nan values

df = df.dropna() # removes rows with any NaN values

df = df.reset_index() # reset's row indexes in case any rows were dropped

df.head()
# Dropping multiple rows

df = df.drop(["Name", "Cabin", "Ticket"], axis=1) # we won't use text features for our initial basic models

df.head()
# Map feature values

df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df["Embarked"] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)

df.head()
# Lambda expressions to create new features

def get_family_size(sibsp, parch):

    family_size = sibsp + parch

    return family_size



df["Family_Size"] = df[["SibSp", "Parch"]].apply(lambda x: get_family_size(x["SibSp"], x["Parch"]), axis=1)

df.head()
# Reorganize headers

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked', 'Survived']]

df.head()
# Saving dataframe to CSV

df.to_csv("processed_titanic.csv", index=False)
# See your saved file

!ls -l