import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]



df = pd.read_csv('../input/auto.csv', names = headers)

df.head()
# replace "?" to NaN

df.replace("?", np.nan, inplace = True)

df.head()
missing_data = df.isnull().sum()

missing_data.sort_values(inplace=True, ascending=False)

missing_data.head()
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)

print("Average of normalized-losses: ", avg_norm_loss)



avg_bore = df['bore'].astype('float').mean(axis=0)

print("Average of bore: ", avg_bore)



avg_stroke = df["stroke"].astype("float").mean(axis = 0)

print("Average of stroke:", avg_stroke)



avg_horsepower = df['horsepower'].astype('float').mean(axis=0)

print("Average horsepower:", avg_horsepower)



avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)

print("Average peak rpm:", avg_peakrpm)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

df["stroke"].replace(np.nan, avg_stroke, inplace = True)

df["bore"].replace(np.nan, avg_bore, inplace=True)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 

df["num-of-doors"].replace(np.nan, "four", inplace=True)
# simply drop whole row with NaN in "price" column

df.dropna(subset=["price"], axis=0, inplace=True)



# reset index, because we droped two rows

df.reset_index(drop=True, inplace=True)

df.head()
df.dtypes
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

df[["price"]] = df[["price"]].astype("float")

df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.dtypes
df.head()
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)

df['city-L/100km'] = 235/df["city-mpg"]



# check your transformed data 

df.head()
# transform mpg to L/100km by mathematical operation (235 divided by mpg)

df["highway-mpg"] = 235/df["highway-mpg"]



# rename column name from "highway-mpg" to "highway-L/100km"

df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)



# check your transformed data 

df.head()
# replace (original value) by (original value)/(maximum value)

df['length'] = df['length'] / df['length'].max()

df['width'] = df['width'] / df['width'].max()

df['height'] = df['height'] / df['height'].max()



# show the scaled columns

df[["length","width","height"]].head()
df["horsepower"] = df["horsepower"].astype(int, copy=True)
plt.hist(df["horsepower"])



# set x/y labels and plot title

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("horsepower bins")
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

bins
group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )

df[['horsepower','horsepower-binned']].head(20)
df["horsepower-binned"].value_counts()
plt.bar(group_names, df["horsepower-binned"].value_counts())



# set x/y labels and plot title

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("horsepower bins")
a = (0,1,2)



# draw historgram of attribute "horsepower" with bins = 3

plt.hist(df["horsepower"], bins = 3)



# set x/y labels and plot title

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("horsepower bins")
df.columns
dummy1 = pd.get_dummies(df["fuel-type"])

dummy1.head()
dummy1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)

dummy1.head()
# merge data frame "df" and "dummy_variable_1" 

df = pd.concat([df, dummy1], axis=1)



# drop original column "fuel-type" from "df"

df.drop("fuel-type", axis = 1, inplace=True)

df.head()
# get indicator variables of aspiration and assign it to data frame "dummy2"

dummy2 = pd.get_dummies(df['aspiration'])



# change column names for clarity

dummy2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)



# show first 5 instances of data frame "dummy1"

dummy2.head()
#merge the new dataframe to the original datafram

df = pd.concat([df, dummy2], axis=1)



# drop original column "aspiration" from "df"

df.drop('aspiration', axis = 1, inplace=True)
df.head()