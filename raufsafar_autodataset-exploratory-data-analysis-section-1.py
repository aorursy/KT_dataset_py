import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = '../input/automobile-dataset/Automobile_data.csv'
df = pd.read_csv(data)

# show the first 2 rows using dataframe.head() method

print("The first 5 rows of the dataframe") 

df.head()
# create headers list

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]

print("headers\n", headers)
# add headers using followingmethod

df.columns = headers

df.head(3)
# we can drop missing values along the column "price" as follows

df.dropna(subset=["price"], axis=0)
# replace "?" to NaN

df.replace("?", np.nan, inplace = True)

df.head(3)
missing_data = df.isnull()

missing_data.head(3)
# Using loop in Python

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('='*50)
avg_norm_loss = df['normalized-losses'].astype('float').mean()

print('Average of normalized losses:', avg_norm_loss)
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace = True)
avg_bore=df['bore'].astype('float').mean()

print("Average of bore:", avg_bore)

df["bore"].replace(np.nan, avg_bore, inplace=True)
#Replace "NaN" by mean value in "stroke" column

avg_stroke=df['stroke'].astype('float').mean()

print('Avg of storke:', avg_stroke)

df['stroke'].replace(np.nan,avg_stroke,inplace =True)
#Calculate the mean value for the 'horsepower' column

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)

print("Average horsepower:", avg_horsepower)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
 #Calculate the mean value for 'peak-rpm' column:

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)

print("Average peak rpm:", avg_peakrpm)

#Replace NaN by mean value:

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
#To see which values are present in a particular column, we can use the ".value_counts()" method:

df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 

df["num-of-doors"].replace(np.nan, "four", inplace=True)
# simply drop whole row with NaN in "price" column

df.dropna(subset =['price'], axis = 0, inplace =True)



# reset index, because we droped two rows

df.reset_index(drop = True, inplace =True)
df.head()
df.dtypes
#Convert data types to proper format

df[['bore','stroke']] = df[['bore','stroke']].astype("float")

df[['normalized-losses']] = df[['normalized-losses']].astype('int')

df["price"] = df["price"].astype("float")

df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
#Let us list the columns after the conversion

df.dtypes
# 1st Method

df['length'] = df['length']/df['width'].max()

df[['width']] = df[['width']] / df[['width']].max()
# Lets do 2nd Method now

df['height'] = df['height']-df['height'].min() / df['height'].max() -df['height'].min()
df[["length","width","height"]].head()
#Convert data to correct format

df["horsepower"]=df["horsepower"].astype(int, copy=True)
#We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

bins
# we set the group names

group_names = ['Low', 'Medium', 'High']
#We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels =group_names, include_lowest=True)

df[['horsepower','horsepower-binned']].head()
df["horsepower-binned"].value_counts()
# Indicator variable (or dummy variable)



# What is an indicator variable?

# An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.



# Why we use indicator variables?



# So we can use categorical variables for regression analysis in the later modules.



# Example

# We see the column "fuel-type" has two unique values, "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.



# We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.
df.columns
dummy_variable_1 =pd.get_dummies(df['fuel-type'])

dummy_variable_1
dummy_variable_1.rename(columns ={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace =True)

dummy_variable_1
# merge data frame "df" and "dummy_variable_1" 

df = pd.concat([df, dummy_variable_1], axis=1)



# drop original column "fuel-type" from "df"

df.drop("fuel-type", axis = 1, inplace=True)
df.head()
dummy_variable_2 = pd.get_dummies(df["aspiration"])

dummy_variable_2.rename(columns={'aspiration':'std', 'aspiration':'turbo'}, inplace=True)

dummy_variable_2.head()
# merge data frame "df" and "dummy_variable_1" 

df = pd.concat([df, dummy_variable_2], axis=1)



# drop original column "fuel-type" from "df"

df.drop("aspiration", axis = 1, inplace=True)
df.head()
df.to_csv('Automobile_data_set')