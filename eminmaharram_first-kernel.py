# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#First we have to read the DataFrame

df=pd.read_csv("/kaggle/input/iris/Iris.csv") 
#After click to df

df 
#Let's look at the top 10 lines

df.head(10) 
#Let's look at the last 10 lines

df.tail(10)
#we can look at the information of df

df.info()
#we can look at the number of columns and rows of the df 

df.shape
#Let's look at column names

df.columns
#Let's look at indexes

df.index
#we can look at the specific information of df

df.describe().T
#Let's look at the number of missing values in each column

df.isnull().sum()
#Let's look at data types

df.dtypes
#Let's look at different values in the Species column

df["Species"].unique()
#Let's look at the number of different values in the Species column

df["Species"].nunique()
#Let's import Seaborn

import seaborn as sns    
#we can look at the "SepalLengthCm","SepalWidthCm", "PetalLengthCm","PetalWidthCm" using the boxplot

sns.boxplot(x=df["SepalLengthCm"]);
#Let's look at the relationship between the Species and the SepalLengthCm

sns.boxplot(x="SepalLengthCm",y="Species",data=df);
#we can look at the relationship between variables and species,using the scatterplot.

sns.scatterplot(x="PetalLengthCm",y="SepalLengthCm",hue="Species",data=df);
#here we removed the Id and updated the df

df.drop(["Id"],axis=1,inplace=True);
#Let's look at df again

df
#Using "pairplot" we can look at the relationship between variables

sns.pairplot(df);
#Using the hue argument, we can look at the relationship of each type with variables

sns.pairplot(df,hue="Species");