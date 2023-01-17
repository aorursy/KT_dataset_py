import numpy as np 

import pandas as pd
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

data = pd.read_csv('../input/imports-85.data.txt', names=cols)

print(data.shape)

data.head()

data = data.replace("?", np.NaN)

data.head()
data.isnull().any().any()
data.isnull().sum()
avg_norm_loss = data['normalized-losses'].astype("float").mean()

avg_norm_loss
data["normalized-losses"].replace(np.NaN, avg_norm_loss, inplace = True)

data["normalized-losses"]
avg_bore = data["bore"].astype("float").mean()

data["bore"].replace(np.NaN, avg_bore, inplace = True)

data['bore']
avg_stroke = data["stroke"].astype("float").mean(axis = 0)

print("Average of stroke:", avg_stroke)



# replace NaN by mean value in "stroke" column

data["stroke"].replace(np.nan, avg_stroke, inplace = True)
data.isnull().sum()
data["bore"].dtype
# for head in cols:

#     if data[head].isnull().any() == True and (data[head].dtype == "int64" or "float64") :

#         avg = data[head].astype("float").mean(axis = 0)

#         print(avg)
avg_horsepower = data['horsepower'].astype('float').mean(axis=0)

print("Average horsepower:", avg_horsepower)

data['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
avg_peakrpm= data['peak-rpm'].astype('float').mean(axis=0)

print("Average peak rpm:", avg_peakrpm)

data['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
data['num-of-doors'].value_counts()
data['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 

data["num-of-doors"].replace(np.nan, "four", inplace=True)

data.head()
# simply drop whole row with NaN in "price" column

before_rows = data.shape[0]

data.dropna(subset=["price"], axis=0, inplace=True)

after_rows = data.shape[0]

print("Number of dropped rows {}".format(before_rows - after_rows))

# reset index, because we droped two rows

data.reset_index(drop=True, inplace=True)
data.shape
data.dtypes
data[["bore", "stroke"]] = data[["bore", "stroke"]].astype("float")

data[["normalized-losses"]] = data[["normalized-losses"]].astype("int")

data[["price"]] = data[["price"]].astype("float")

data[["peak-rpm"]] = data[["peak-rpm"]].astype("float")

data.head()
data.head()
# transform mpg to L/100km by mathematical operation (235 divided by mpg)

data["highway-mpg"] = 235/data["highway-mpg"]



# rename column name from "highway-mpg" to "highway-L/100km"

data.rename(columns = {'highway-mpg':'highway-L/100km'}, inplace=True)



# check your transformed data 

data.head()
# replace (original value) by (original value)/(maximum value)

data['length'] = data['length']/data['length'].max()

data['width'] = data['width']/data['width'].max()

data.head()
# replace height with normalized values

data["height"] = data["height"]/data["height"].max()
data.head()
data["horsepower"] = data["horsepower"].astype(int, copy = True)
data.head()
%matplotlib inline

import matplotlib.pyplot as plt

plt.hist(data["horsepower"])



# set x/y labels and plot title

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("horsepower bins")
bins = np.linspace(min(data["horsepower"]), max(data["horsepower"]), 4)

bins
group_names = ['Low', 'Medium', 'High']
data['horsepower-binned'] = pd.cut(data['horsepower'], bins, labels=group_names, include_lowest=True )

data[['horsepower','horsepower-binned']].head(20)
data["horsepower-binned"].value_counts()
%matplotlib inline 

import matplotlib.pyplot as plt

plt.bar(group_names, data["horsepower-binned"].value_counts())



# set x/y labels and plot title 

plt.xlabel('Horsepower')

plt.ylabel("Count")

plt.title("Horsepower Bins")
%matplotlib inline

import matplotlib.pyplot as plt



a = (0, 1, 2)



# draw histogram of attribute 

plt.hist(data["horsepower"], bins = 3)



# set x / y labels and plot title 

plt.xlabel("horsepower")

plt.ylabel("count")

plt.title("Horsepower Bins")

plt.show()
data.columns
dummy_variable_1 = pd.get_dummies(data["fuel-type"])

dummy_variable_1.head()
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)

dummy_variable_1.head()
# merge data frame "df" and "dummy_variable_1" 

data = pd.concat([data, dummy_variable_1], axis=1)



# drop original column "fuel-type" from "df"

data.drop("fuel-type", axis = 1, inplace=True)
# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"

dummy_variable_2 = pd.get_dummies(data['aspiration'])



# change column names for clarity

dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)



# show first 5 instances of data frame "dummy_variable_1"

dummy_variable_2.head()
#merge the new dataframe to the original datafram

data = pd.concat([data, dummy_variable_2], axis=1)



# drop original column "aspiration" from "df"

data.drop('aspiration', axis = 1, inplace=True)
data.head(10)
data.describe()
# Convert to CSV file

data.to_csv('wrangled_data.csv')