import pandas as pd
import numpy as np
# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users_2.csv')
#Preview the training data
train_users.head()
print("There are", train_users.shape[0], "users in the training dataset.")
# Let's see the unique values of the categorical variables.

# There's something peculiar going on in the AGE field.
# Let's look more closely.
for x in pd.unique(train_users['age']):
    print("{:.0f}".format(x))
