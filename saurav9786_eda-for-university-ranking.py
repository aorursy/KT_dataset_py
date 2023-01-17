# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the csv file

unv_df=pd.read_csv("/kaggle/input/world-university-rankings/cwurData.csv")
#displaying the top 10 data

unv_df.head(10)
#Calculate the no of rows and columns

unv_df.shape
unv_df.dtypes
#Check for the missing values 



unv_df.isna().any()
#Calculating the no of missing values for broad_impact column

unv_df['broad_impact'].isna().sum()
unv_df.nunique()
unv_df["year"].value_counts()
unv_df_2012=unv_df.loc[unv_df['year'] == 2012]

unv_df_2012.head(10)
#Shape of the data



unv_df_2012.shape
#Missing values in the columns



unv_df_2012.isna().any()
#Count the missing values in the broad_impact data set



unv_df_2012.isna().sum()
#Drop the broad_impact column as the entire data is missing for all the rows 



unv_df_2012.drop("broad_impact",axis=1,inplace=True)
#List of countries in the dataset



unv_df_2012["country"].unique()
unv_df_2012.describe().T
sns.pairplot(unv_df_2012)
# Correlation with heat map

import matplotlib.pyplot as plt

import seaborn as sns

corr =unv_df_2012.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#Top ten universities in terms of world rank 



unv_df_2012_new= unv_df_2012.head(10)

plt.scatter(unv_df_2012_new.country,unv_df_2012_new.institution)

plt.xlabel('countries')

plt.ylabel('universities')

plt.title('rank')

plt.show()
#Top ten universities of the USA as per the national rankings



unv_df_2012_USA=unv_df_2012.loc[unv_df_2012['country']=="USA"]



unv_df_2012_USA.sort_values('national_rank')

print(unv_df_2012_USA["institution"].head(10))
# University of USA world ranking 

plt.hist(unv_df_2012_USA.world_rank,bins=2) #histogram

plt.title('Universities of USA')

plt.xlabel("rank of institutions")

plt.show()
#Alumni employed by the top ten universities  



plt.figure(figsize=(25,7)) # this creates a figure 20 inch wide, 7 inch high

sns.barplot(unv_df_2012_new['institution'], unv_df_2012_new['alumni_employment'])

#Impact of score to the 

plt.figure(figsize=(25,7)) # this creates a figure 20 inch wide, 7 inch high

sns.barplot(unv_df_2012_new['institution'], unv_df_2012_new['score'])