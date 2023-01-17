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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

df = pd.read_csv("../input/dpt2018.csv", sep=";")
#Let's read the first 10 rows of the dataframe

df.head(10)
#Checking the structure of my dataframe (as we'll see, there are a lot of rows)

df.shape
#Checking the column names, to avoid bad surprises like white spaces in front of a column name

df.columns
#Changing columns names,as to be more intuitive, and not in French :)

df.columns=['sex', 'name','year','departement','number']

df.head(5)
#Checking the null values

df.isnull().count()
#Checking types of objects in my dataframe

df.dtypes
#Changing the XXXX values into 0000, we want numeric characters

df.loc[df["year"]=="XXXX", "year"]="0000"
#Since we know we'll need a barplot, let's change the year type into an integer

df['year'] = df['year'].astype(int)
#As we can see, there are no null values, but year XXXX is not a useful information. Let's try and lock the XXXX values and count them

df.loc[df["year"]==0000].count()
#There are few 0000 values in the year column, compared to the global number of rows. We can get rid of them,they represent 1% of the dataframe

df = df[df.year !=0000]
#Let's try and inspect some aggregate functions in Pandas

df.groupby('name').count().sort_values('year', ascending=False).head(10)

#We can see that since there are mostly string type objects, it's going to be difficult tu use aggregate functions.
#Trying to sort values, using two important parameters : year and sex

df.sort_values(by=["year"]).sort_values("sex")

#Not so concluant afterall...
#Let's now try and explore otherwise, adding new conditions to our request, namely year, sex and name.Let's order them by the quantity

df.groupby(["year", "sex", "name"]).sum().reset_index().sort_values("number", ascending=False)

#Now, we're kind of getting somewhere!
#Let's then just create the function that will return us : girls and boys most frequent names in France per year. I will only ask the first 5 names, as I know I want to plot them and it needs to be easy to read

def names (df, year):

  df=df.groupby(["year", "sex", "name"]).sum().reset_index().sort_values("number", ascending=False) 

  return pd.concat([df.loc[(df.year==year)&(df.sex==1)][:5],df.loc[(df.year==year)&(df.sex==2)][:5]])
#Let's test our function with the best year ever :)

names(df, 1980)
#OK, now let's plot to see it clearly. 

df1980=names(df, 1980) #creating a new dataframe to plot. I took 1980, but you can chose a year of your own. 

plt.figure(figsize=(15, 10))

plt.title("MOST FREQUENT NAMES IN 1980")

plt.ylabel('Number')

plt.xlabel('Names')

sns.barplot(data=df1980, x=df1980["name"], y=df1980["number"], hue=df1980["sex"], palette="husl")