import os

import subprocess

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import matplotlib.pyplot as plt

%matplotlib inline
def encode_target(df, target_column):

    """Add column to df with integers for the target.



    Args

    ----

    df -- pandas DataFrame.

    target_column -- column to map to int, producing

                     new Target column.



    Returns

    -------

    df_mod -- modified DataFrame.

    targets -- list of target names.

    """

    df_mod = df.copy()

    targets = df_mod[target_column].unique()

    map_to_int = {name: n for n, name in enumerate(targets)}

    df_mod["Target"] = df_mod[target_column].replace(map_to_int)



    return (df_mod, targets)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Train data using decision trees
#Get rid of Nan and fill them with an appropiate value

df_train_new=df_train.copy()

df_train_new['Age']=df_train_new.Age.fillna(0)

df_train_new['Cabin']=df_train_new.Cabin.fillna('Unknown')

df_train_new['Embarked']=df_train_new.Embarked.fillna('Unknown')

#make a new dataframe df2 which map string to integers

df2, targets = encode_target(df_train_new, "Name")

df2['Name']=df2.Target

df2, targets = encode_target(df2, "Sex")

df2['Sex']=df2.Target

df2, targets = encode_target(df2, "Ticket")

df2['Ticket']=df2.Target

df2, targets = encode_target(df2, "Cabin")

df2['Cabin']=df2.Target

df2, targets = encode_target(df2, "Embarked")

df2['Embarked']=df2.Target
#consinder which feature to use for dt

features=['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

print("* features:", features, sep="\n")
#just to check if we miss something

print(df2[features].shape)

print(df2[features].dropna().shape)
#Calculate the dt

y = df2["Survived"]

X = df2[features]

dt = DecisionTreeClassifier(min_samples_split=2, random_state=99)

dt.fit(X, y)
#Test the data
#Get rid of Nan and fill them with an appropiate value

df_test_new=df_test.copy()

df_test_new['Age']=df_test_new.Age.fillna(0)

df_test_new['Fare']=df_test_new.Fare.fillna(0)

df_test_new['Cabin']=df_test_new.Cabin.fillna('Unknown')

df_test_new['Embarked']=df_test_new.Embarked.fillna('Unknown')

#make a new dataframe df3 which map string to integers

df3, targets = encode_target(df_test_new, "Name")

df3['Name']=df3.Target

df3, targets = encode_target(df3, "Sex")

df3['Sex']=df3.Target

df3, targets = encode_target(df3, "Ticket")

df3['Ticket']=df3.Target

df3, targets = encode_target(df3, "Cabin")

df3['Cabin']=df3.Target

df3, targets = encode_target(df3, "Embarked")

df3['Embarked']=df3.Target
#to predict the test

Predicted_test=dt.predict(df3[features])
df_test_new['Survived_predicted']=Predicted_test
df_test['Survived_predicted']=Predicted_test
df_test.Survived_predicted.value_counts()
df_test.to_csv('predicted_survived_test.csv')