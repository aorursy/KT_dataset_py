# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#create dataframe

url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

df = pd.read_csv(url, header = None)



# create headers list

headers_names = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]



#assign headers list to dataframe

df.columns = headers_names



df.head(10)
#dataset exploration

df.dtypes
#dataset exploration

df.describe(include='all')
#dataset exploration

df.info()
#to replace '?' with NaN so that later dropna() can remove the missing values

df.replace('?', np.nan, inplace = True)

df.head()
#checking to see if we have any missing values

missing_data = df.isnull()

missing_data.head(5)
#looping through the missing values to see distinct count

for cols in missing_data.columns.values.tolist():

    print(cols)

    print(missing_data[cols].value_counts())

    print("")
#"normalized-losses": 41 missing data, replace them with mean

avg_normalized_losses = df['normalized-losses'].astype('float').mean(axis=0)

df['normalized-losses'].replace(np.nan, avg_normalized_losses, inplace = True)

df.head()
#"stroke": 4 missing data, replace them with mean

avg_stroke = df['stroke'].astype('float').mean(axis=0)

df['stroke'].replace(np.nan, avg_stroke, inplace = True)

df.head()
#"bore": 4 missing data, replace them with mean

avg_bore = df['bore'].astype('float').mean(axis=0)

df['bore'].replace(np.nan, avg_bore, inplace = True)

df.head()
#"horsepower": 2 missing data, replace them with mean

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)

df['horsepower'].replace(np.nan, avg_horsepower, inplace = True)

df.head()
#"peak-rpm": 2 missing data, replace them with mean

avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace = True)

df.head()
#to see num-of-doors

df['num-of-doors'].value_counts()

# We can see that four doors are the most common type.

#"num-of-doors": 2 missing data, replace them with "four". 

# Reason:  84% sedans is four doors. Since four doors is most frequent, it is most likely to occur.

df["num-of-doors"].replace(np.nan, "four", inplace = True)

df["num-of-doors"].head(10)
#to drop missing values along the column "price" as price is our output column and we can't have missing values in the output column.

df.dropna(subset=["price"], axis=0, inplace = True)



df.reset_index(drop = True, inplace = True)

df.head(10)
#to verify the column data types

df.dtypes
#Convert data types to proper format

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

df[["price"]] = df[["price"]].astype("float")

df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")



df.dtypes
# Transform mpg to L/100km:

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)

df['city-L/100km'] = 235 / df["city-mpg"]

df.head()
# transform mpg to L/100km in the column of "highway-mpg" as well

df['highway-L/100km'] = 235/df["highway-mpg"]

df.head()
#to scale the columns "length", "width" and "height", would like to Normalize those variables so their value ranges from 0 to 1.

# replace (original value) by (original value)/(maximum value)

df['length'] = df['length'] / df['length'].max()

df['width'] = df['width'] / df['width'].max()

df['height'] = df['height'] / df['height'].max()
#to segment the 'horsepower' column into 3 bins 'Little horsepower', 'Medium horsepower' and 'High horsepower'

df["horsepower"]=df["horsepower"].astype(int, copy=True)



%matplotlib inline

import matplotlib as plt

from matplotlib import pyplot



plt.pyplot.hist(df["horsepower"])



#set x,y label and set title

plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("count")

plt.pyplot.title("horsepower bins")
#create three bins

bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)

bins
group_names = ["Low", "Medium", "High"]



df["horsepower-bins"] = pd.cut(df["horsepower"], bins, labels = group_names, include_lowest = True)

df[["horsepower", "horsepower-bins"]].head(10)
#number of vehicles in each bin

df["horsepower-bins"].value_counts()
#plot the distro of each bin

plt.pyplot.bar(group_names, df["horsepower-bins"].value_counts())



plt.pyplot.xlabel("horsepower")

plt.pyplot.ylabel("counts")

plt.pyplot.title("horsepower bins")

# convert "fuel-type" into indicator variables.

dummy_variable_1 = pd.get_dummies(df["fuel-type"])

dummy_variable_1.head()



dummy_variable_1.rename(columns={"gas":"fuel-type-gas", "diesel":"fuel-type-diesel"}, inplace = True)

dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis = 1)

df.head()



df.drop("fuel-type", axis = 1, inplace = True)

df.head()
dummy_variable_2 = pd.get_dummies(df['aspiration'])

dummy_variable_2.head()

dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace = True)

dummy_variable_2.head()
df = pd.concat([df, dummy_variable_2],axis=1)

df.drop("aspiration", axis=1, inplace=True)

df.head(10)