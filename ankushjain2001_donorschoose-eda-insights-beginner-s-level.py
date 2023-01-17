import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
df_donors = pd.read_csv('../input/Donors.csv')
df_donations = pd.read_csv('../input/Donations.csv')
df_teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
df_projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False)
df_resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
df_schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)

# SMALL DATAFRAMES
df_donors_sm = df_donors.iloc[1:200,]
df_donors.head()
df_donors.describe()
df_donations.head()
df_donations.describe()
df_teachers.head()
df_teachers.describe()
df_projects.head()
print("Total Rows of Data :",len(df_projects))
print(df_projects["Project Type"].describe())
print(df_projects["Project Subject Category Tree"].describe())
print(df_projects["Project Grade Level Category"].describe())
print(df_projects["Project Resource Category"].describe())
print(df_projects["Project Current Status"].describe())
print(df_projects.describe())
df_resources.head()
print(df_resources["Resource Vendor Name"].describe())
print(df_resources.describe())
df_schools.head()
print(df_schools["School Metro Type"].describe())
print(df_schools["School State"].describe())
print(df_schools.describe())
df_donations_donors = df_donations.merge(df_donors, on='Donor ID', how='inner')
df_donations_donors.head()

sns.countplot(y='Donor State', data=df_donors_sm, color='c', order=pd.value_counts(df_donors['Donor State']).iloc[:10].index);
