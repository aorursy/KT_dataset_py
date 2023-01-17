# Importing Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



import warnings

warnings.filterwarnings('ignore')



#for displaying 500 results in pandas dataframe

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# File Path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('/kaggle/input/2020-democratic-primary-endorsements/endorsements-2020.csv')

df.head()
#Shape of dataframe

print(" Shape of dataframe: ", df.shape)

# names of columns

print(df.columns)
#Datatypes

print(df.info())

#Unique values



print(len(df.position.unique()))

print(len(df.city.unique()))
#Let's see how null values look in dataframe

#Missing data as white lines 

import missingno as msno

msno.matrix(df,color=(0,0.3,0.9))
#Null values



null= df.isnull().sum().sort_values(ascending=False)

total =df.shape[0]

percent_missing= (df.isnull().sum()/total).sort_values(ascending=False)



missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])



missing_data.reset_index(inplace=True)

missing_data= missing_data.rename(columns= { "index": " column name"})

 

print ("Null Values in each column:\n", missing_data)

#drop columns



df= df.drop(columns= ["date", "city", "body", "order", "district", "source", "endorsee"])
#Dealing with Null values in endorser party



df["endorser_partyna"]= df["endorser party"]

df["endorser_partyna"]=df["endorser_partyna"].fillna("Missing")
df["endorser_partyna"].value_counts()
df.head()
sns.boxplot(df["points"])
sns.violinplot(x="endorser_partyna", y="points", data=df, size=6)
df.groupby("category").points.sum()
df.groupby("position").points.sum()
df.groupby("endorser_partyna").points.sum()
df.columns
df.groupby(["category", "position"]).points.sum()
df.state.value_counts(normalize= True)
df.columns
df.groupby(["category", 'endorser_partyna']).points.sum()
df.groupby(["category", "position", 'endorser_partyna']).points.sum()