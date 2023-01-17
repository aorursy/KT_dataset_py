# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required libraries for EDA

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt     #visualisation

import seaborn as sns               #visualisation



# Loading the data into the data frame 

df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')



# To display the top 5 rows 

df.head(5)



#columns 

col = df.columns



for var in col:

    print(var)

    

# Checking the types of data

df.dtypes





#Find null percentages

df.isnull().mean()*100



#Seperate numerical and categorical 



#Identifying Numerical Variables

num_var = ['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']

df['salary'].mean()

df['ssc_p'].median()



#Identifying Categorical Variables

cat_var = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']

df['gender'].mode().iloc[0]

df['ssc_b'].mode().iloc[0]



# removing null values



df.isnull().mean()



# only salary has null values then we can only do imputation for num_var

def impute_num(df,variable):

    return df[variable].fillna(df[variable].mean())



for var in num_var:

    print(var)

    df[var]=impute_num(df,var)

    

df.isnull().mean()



# plotting boxplot

sns.boxplot(df['salary'])
# remove outliners

df_out = df[['salary']]



lb = df_out.quantile(0.1) #lb = lowerboundary

ub = df_out.quantile(0.9) #ub = upperboundary



df_out = df_out.clip(lower = df_out.quantile(0.1),upper = df_out.quantile(0.9),axis=1)

    

df = df.drop(['salary'], axis=1)



df = pd.concat([df,df_out],axis=1,join='inner')



sns.boxplot(df['salary']) 
#Removing unneccessary columns

df = df.drop(['sl_no'], axis=1) 



#creating dummies



dummy1 = pd.get_dummies(df['specialisation'])

dummy2 = pd.get_dummies(df['workex'])

dummy3 = pd.get_dummies(df['degree_t'])

dummy4 = pd.get_dummies(df['hsc_s'])

dummy5 = pd.get_dummies(df['status'])



df = pd.concat([df,dummy1,dummy2,dummy3,dummy4,dummy5],axis=1)



df.head()
#correlation

c=df.corr()

c
# Heatmap

plt.figure(figsize=(14,14))

sns.heatmap(c, cmap='BrBG', annot=True)
#Visualization



# SSC board vs Status

print(sns.countplot(df['ssc_b'], hue = df['status']))
# Work experience vs Status

print(sns.countplot(df['workex'], hue = df['status']))
# Specialisation vs Status

print(sns.countplot(df['specialisation'], hue = df['status']))
# Degree type vs Status

print(sns.countplot(df['degree_t'], hue = df['status']))