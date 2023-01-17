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
train_users = pd.read_csv('../input/train_users.csv')
test_users = pd.read_csv('../input/test_users.csv')
sess_users = pd.read_csv('../input/sessions.csv') 
country_users = pd.read_csv('../input/countries.csv') 
age_gender = pd.read_csv('../input/age_gender_bkts.csv')
print ("train_users.csv\n", list(train_users))
print ("session.csv\n", list(sess_users))
print ("countries.csv\n", list(country_users))
print ("age_gender_bkts.csv\n", list(age_gender))
