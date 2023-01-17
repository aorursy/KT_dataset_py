# Import libraries and data

import numpy as np

import pandas as pd

import scipy

import matplotlib.pyplot as plt



df = pd.read_csv('../input/ab-testing/ab_data.csv')



df.head(20)
# Checking to see if there are any users in control that saw new page and users in treatment that saw old page

df.groupby(['group','landing_page']).count()
# There seems to bee a mistake in inputs where some of the control group saw the new page and some of the treatment group saw the old page.

# Since we're not sure which way to switch it to, we're going to remove it
# Removing control/new_page and treatment/old_page

df_cleaned = df.loc[(df['group'] == 'control') & (df['landing_page'] == 'old_page') | (df['group'] == 'treatment') & (df['landing_page'] == 'new_page') ]
df_cleaned.groupby(['group','landing_page']).count()
# Checking for duplicate values

df_cleaned['user_id'].duplicated().sum()
# Finding user_id for duplicate value

df_cleaned[df_cleaned.duplicated(['user_id'],keep=False)]['user_id']
df[df['user_id'] == 773192]
df_cleaned = df.drop_duplicates(subset='user_id', keep="first")

df_cleaned['user_id'].duplicated().sum()
groups = df_cleaned.groupby(['group','landing_page','converted']).size()

groups.plot.bar()
df['landing_page'].value_counts().plot.pie()
### Re-arrrange data into 2x2 for Chi-Squared



# 1) Split groups into two separate DataFrames

a = df[df['group'] == 'control']

b = df[df['group'] == 'treatment']



# 2) A-click, A-noclick, B-click, B-noclick

a_click = a.converted.sum()

a_noclick = a.converted.size - b.converted.sum()

b_click = b.converted.sum()

b_noclick = b.converted.size - b.converted.sum()



# 3) Create np array

T = np.array([[a_click, a_noclick], [b_click, b_noclick]])
import scipy

from scipy import stats



print(scipy.stats.chi2_contingency(T,correction=False)[1])
a_CTR = a_click / (a_click + a_noclick)

b_CTR = b_click / (b_click + b_noclick)

print(a_CTR, b_CTR)