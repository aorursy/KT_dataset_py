# https://www.kaggle.com/donorschoose/io/data

%matplotlib inline
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

%matplotlib inline

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

import os
# print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

donations = pd.read_csv('../input/Donations.csv')
print(donations.shape)
donations.head()
# many 0 donations? 
donations["Donation Amount"].describe()
pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID',nrows=6)
pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID",nrows=3)
pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID",nrows=3)
pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID',nrows=4)
resources = pd.read_csv('../input/Resources.csv', index_col="Project ID",error_bad_lines=False,warn_bad_lines = False)
print(resources.shape)
resources["sum_resource_price"] = resources["Resource Quantity"]*resources["Resource Unit Price"]
resources.head()
# resources.describe(include="all")
# donors = pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID')
# projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID")
# schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID")

# projects_light = projects.drop(columns='Project Essay',axis=1)
df = donations.join(pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,index_col="Project ID"),on='Project ID',how='left')
df = df.join(pd.read_csv('../input/Donors.csv', low_memory=False,index_col='Donor ID'),on='Donor ID')
df = df.join(pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False,index_col="School ID"),on='School ID')
df = df.join(pd.read_csv('../input/Teachers.csv', low_memory=False,index_col='Teacher ID'),on='Teacher ID')

# df = df.join(resources,on="Project ID") # multiple resources per project
print(df.shape, "\n")
df.describe()
print(df.shape)
df.head()
df = df.loc[df["Donation Amount"]>5]
df.shape
df["count_teacher"] = df.groupby('Teacher ID')['Project ID'].transform("count")
df["count_donor"] = df.groupby('Donor ID')['Project ID'].transform("count")
df["count_school"] = df.groupby('School ID')['Project ID'].transform("count")

df[["count_teacher","count_donor","count_school"]].describe()
df_multidonor = df.loc[df["count_school"]>2]
print(df_multidonor.shape)

df_multidonor = df_multidonor.loc[df_multidonor["count_teacher"]>2]
print(df_multidonor.shape)

df_multidonor = df_multidonor.loc[df_multidonor["count_donor"]>2]
print(df_multidonor.shape)

df_multidonor[["count_teacher","count_donor","count_school"]].describe()
df_multidonor["Donation Amount"].describe()
print(resources.shape)
# resources.loc[resources["Project ID"].isin(df_multidonor["Project ID"])].shape
resources.loc[resources.index.isin(df_multidonor["Project ID"])].shape
df_multidonor.to_csv("merged_donorsChoose-multiDonor3_v1.csv.gz",compression="gzip")
resources.loc[resources.index.isin(df_multidonor["Project ID"])].to_csv("resources_donorsChoose-multiDonor3_v1.csv.gz",compression="gzip")
# df.to_csv("merged_donorsChoose_v1.csv.gz",compression="gzip")