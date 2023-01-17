import csv

import numpy as np

import pandas as pd

import tensorflow as tf



#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# We see that 2 passengers embarked data is missing, we fill those in as the most common port, S

train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'



# First we convert some inputs into ints so we can perform mathematical operations

train['gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train['emb'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)



# There are passengers with missing age data, so we need to fill that in

# We want to get median ages by gender and class

classes = [1, 2 ,3]

genders = [0, 1]

for _class in classes:

    for gender in genders:

        # We locate every passenger of given class and gender whose age is null, and assign it to the median age

        median_age = train[(train.Pclass==_class) & (train.gender==gender)].Age.dropna().median()

        train.loc[(train.Age.isnull()) & (train.gender==gender) & (train.Pclass==_class), 'Age'] = median_age



# Now we have all the data we want. We will not do any feature engineering, 

# as we want the neural net to do that for us

# Now we will format the data so we can use it in tensorflow

train_labels = train.Survived.values

train_dataset = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Survived'], axis=1).values
valid_dataset = dataset[:267:, ::]

train_dataset = dataset[267::, ::]

valid_labels = labels[:267]

train_lables = labels[267:]
test.loc[train.Age.isnull(), 'Embarked'] 
train_dataset.shape[1]

 
 
 

    