# Importing necessary packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.model_selection import train_test_split

# List all files under the input directory to check that our datasets are there
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading the datasets
df_pastLoans = pd.read_csv('../input/lending-game/PastLoans_2.csv')
df_loansApplications = pd.read_csv('../input/lending-game/LoanApplications_2.csv')
df_pastLoans.head()
df_loansApplications.head()
# Basic stats
df_loansApplications.describe()
# NA search
df_pastLoans.isnull().sum()
# Number of loan applicants with 0 income
cnt = 0
for i in range(0, len(df_loansApplications['income'])): 
    if df_loansApplications['income'][i] == 0:
        cnt = cnt + 1
cnt
# There seams to be almost 3 times more male borrowers than females
df_loansApplications['sex'].value_counts()
# There are more borrowers that are married
df_loansApplications['marital'].value_counts()
# Function to compute overall probability of default in a df
def compute_probability_of_default(df):
    defaults_nb = df['default'].value_counts()[1]
    non_defaults_nb = df['default'].value_counts()[0]
    proba = defaults_nb / (defaults_nb + non_defaults_nb)
    print("Probability of default is: " + str(proba*100) + "%")
compute_probability_of_default(df_pastLoans)
# Inspecting if the borrowers with 0 income have a higher probability of default
has_0_income = df_pastLoans['income']==0
has_0_friends = df_pastLoans['employment'] == 'student'
df_0_income_pastloans = df_pastLoans[has_0_income]
df_0_income_pastloans_and_friends = df_0_income_pastloans[has_0_friends]
compute_probability_of_default(df_0_income_pastloans)
# Inspecting if the borrowers with 0 fb friends default more
has_0_fb_friends = df_pastLoans['facebook']==0
df_0_fb_friends_pastloans = df_pastLoans[has_0_fb_friends]
compute_probability_of_default(df_0_fb_friends_pastloans)
# Inspecting how high income influences the probability of default
has_high_income = df_pastLoans['income'] > 17414
df_high_income_pastloans = df_pastLoans[has_high_income]
compute_probability_of_default(df_high_income_pastloans)
# Inspecting how middle income influences the probability of default
has_high_income = df_pastLoans['income'] > 5994
df_high_income_pastloans = df_pastLoans[has_high_income]
compute_probability_of_default(df_high_income_pastloans)
df_pastLoans.corr()
# Add 0 income as a flag
df_pastLoans['has_0_income'] = has_0_income
df_pastLoans.head()
# Replacing 0 Facebook friends with mean
mean_fb_friends = df_pastLoans['facebook'].mean()
df_pastLoans['facebook'] = df_pastLoans.facebook.mask(df_pastLoans.facebook == 0, mean_fb_friends)
df_pastLoans.head()
# Replacing 0 income with mean
mean_income = df_pastLoans['income'].mean()
df_pastLoans['income'] = df_pastLoans.income.mask(df_pastLoans.income == 0, mean_income)
df_pastLoans.head()
# Get dummies for employment
df_pastLoans = df_pastLoans.join(pd.get_dummies(df_pastLoans['employment']))
df_pastLoans = df_pastLoans.drop(['employment'], axis=1)
df_pastLoans.head()              
# Split the dataset into X & y (where y is the default to predict)
df_X = df_pastLoans.drop(['default'], axis=1)
df_y = df_pastLoans['default']
# Split test & train randomly
X_train, X_test, y_train, y_test = train_test_split(
     df_X, df_y, test_size=0.2, random_state=42)
import sklearn.linear_model

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, y_train)
clf.predict(X_train.head(300))
# Use score method to get accuracy of model
score = clf.score(X_test, y_test)
print(score)
