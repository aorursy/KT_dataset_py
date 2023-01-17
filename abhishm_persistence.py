import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
%matplotlib inline
# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users_2.csv')
# Load the data into DataFrames
age_gender = pd.read_csv('../input/age_gender_bkts.csv')
countries = pd.read_csv('../input/countries.csv')

session = pd.read_csv('../input/sessions.csv')
train_users.columns
# Are all the users in the training data are unique?
print(np.unique(train_users.id).shape)
print(train_users.shape[0])
