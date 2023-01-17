import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users_2.csv')
test_users = pd.read_csv('../input/test_users.csv')
train_users.head()
print(train_users.shape[0])

print(test_users.shape[0])
# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
users.head()
users.describe()
# Remove ID's since now we are not interested in making predictions
users.drop('id',axis=1, inplace=True)
train_users = pd.read_csv('../input/age_gender_bkts.csv')
print(train_users)