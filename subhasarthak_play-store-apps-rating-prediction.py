# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

pd.set_option('display.max_rows', 100)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

data.shape
print(f"Number of unique apps:{data['App'].nunique()}")
# Since the number of unique apps < Total number of data points, then there must be some duplicate apps.

# I am going to consider Apps with same name and category as the duplicate apps. Let's find out the duplicate apps.



# So there are around 750 apps with the same name and category. So in the next step I am going to remove those

# apps.

data[data.duplicated(subset=["App", "Category"])]["App"].nunique()
# Removing the Duplicates

print(f"Before removing the duplicates, shape of data: {data.shape}")

data= data.drop_duplicates(subset=["App", "Category"], keep='last')

data.reset_index(drop=True, inplace=True)

print(f"After removing the duplicates, shape of data: {data.shape}")
# Let's check the different categories of apps

data["Category"].unique()
# One app has category 1.9 which doesn't make sense.

data[data["Category"] == '1.9']
# It looked like the values are shifted backwards from the Rating column. For now removing the row.

data = data.drop(data[data["Category"] == "1.9"].index)

print(data.shape)
# Missing values in different columns

data.isnull().sum()
# Data Cleaning

# Reference: https://www.kaggle.com/sabasiddiqi/google-play-store-apps-data-cleaning

"""

Size Column

-----------

Size column contains size of the applications - with kb or mb attached to it. So we are going to replace it with

their actual size k - 'e+3', M - 'e+6'. For now we are going to replace the "varies with device part" with NaN 

values which we can fill with the median size for that category.

"""

data["Size"] = data["Size"].str.replace('k', 'e+3')

data["Size"] = data["Size"].str.replace('M', 'e+6')

data["Size"].replace("Varies with device", np.nan, inplace=True)



data["Size"] = pd.to_numeric(data["Size"])
# Let's impute the varies with device value in the column Size with the median value of the "Size" for that 

# particular category of apps

def impute_median(s):

    return s.fillna(s.median())



by_category = data.groupby(["Category"])

data["Size"] = by_category["Size"].transform(impute_median)
"""

Rating Column

-------------

Since rating column contains a lot of missing values, I am going to the fill those with the median Rating

for the apps of the same group as I have done for the Size column.

"""

data["Rating"] = by_category["Rating"].transform(impute_median)
"""

Installs Column

---------------

The Installs column is of type String. But it should be a numeric column as it shows the number of installs.

For that I am going to remove the '+' and ',' from the Installs column.

"""

data["Installs"] = pd.to_numeric(data["Installs"].str.strip("+").str.replace(",", ""))
# As we know the apps with 0 installs should not have any ratings ;) but in this approach we are going to rate

# those apps also which is incorrect. So I am going to modify their ratings to 0 which seems logical & 

# fortunately we don't have any App with negative review.

data.loc[data["Installs"] == 0, ["Rating"]] = 0
"""

Type Column

-----------

Let's checkout the type column which is a categorical column. It has only one missing value. Let's check for which

app we have missing values. 



"Price" of the app for which the "Type" column is missing is 0. Since this is only a single value missing, I am 

going to directly impute it as of "Type" = "Free".

"""

# print(data[data["Type"].isnull()])

data.at[8052, "Type"] = "Free"
"""

Price Column

------------

Price column do not have any missing values. But the prices are present in strings. So first let's remove those and 

then convert it to a numeric column.

"""

data["Price"] = pd.to_numeric(data["Price"].str.strip("$"))
"""

Reviews Column

--------------

Reviews column contains number in string format. So need to convert that to numeric format.

"""

data["Reviews"] = pd.to_numeric(data["Reviews"])
"""

Genres Column

-------------

Since in the "Genres" column some data is present in the a;b format. 

So spliting this columns to 2 different column.

"""

data["Subgenres"] = data["Genres"].apply(lambda x: x.split(';')[-1])

data["Genres"] = data["Genres"].apply(lambda x: x.split(';')[0])

data.head()
"""

Last Updated Column

-------------------

Convert the last updated column to datetime column. Create a new column which indicates the difference between the

last updated column and current date.

"""

data["Last Updated"] = pd.to_datetime(data["Last Updated"])

data["Updategaps"] = pd.to_datetime(datetime.today().strftime("%m-%d-%Y")) - data["Last Updated"]

data.head()
data["Updategaps"] = pd.to_numeric(data["Updategaps"].astype(str).str[:3])
# Removing the "Current Ver", Android Ver", "Last Updated" columns

data.drop(columns=["Current Ver", "Android Ver", "Last Updated"], inplace=True)
def return_plot(figsize=(16, 8), nrows=1, ncols=1):

    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

    return fig, ax
# Let's checkout the Ratings distribution. 

# From the ratings distribution it is clear that most of the apps have rating more than 4. 

fig, ax = return_plot()

sns.distplot(data["Rating"], bins=30, ax=ax)

ax.axvline(data["Rating"].mean(), color="k", linestyle="dashed", linewidth=1)
# Let's check the categories column.

fig, ax = return_plot()

sns.countplot(x="Category", data=data, order=data["Category"].value_counts().index, ax=ax)

plt.xticks(rotation=90)
# Let's look at the type of Apps (Free or paid)

# There are a lot of free apps than paid apps. So can we consider "Free apps" are more popular than "Paid apps" ?

fig, ax = return_plot()

sns.countplot(x="Type", data=data, order=data["Type"].value_counts().index, ax=ax)

plt.xticks(rotation=90)
# Let's drill down the paid apps

paid_apps = data[data["Type"] == "Paid"]

free_apps = data[data["Type"] == "Free"]
fig, ax = return_plot()

sns.countplot(x="Category", data=paid_apps, order=paid_apps["Category"].value_counts().index, ax=ax)

plt.xticks(rotation=90)
# Price distribution of the paid apps

fig, ax = return_plot()

sns.distplot(paid_apps["Price"], bins=50, ax=ax)
# Let's check the apps whose price > 300$

paid_apps[paid_apps["Price"] > 300]
fig, ax = return_plot()

sns.distplot(paid_apps["Rating"], hist=False, ax=ax, label="Paid Apps Rating")

sns.distplot(free_apps["Rating"], hist=False, ax=ax, label="Free Apps Rating")

plt.legend()
# Rating variation with "Type" of app

fig, ax = return_plot()

sns.boxplot(x="Type", y="Rating", data=data, ax=ax)
# Let's check the top 20 apps (Depending on the number of installs)

top_20_installed_apps = data.nlargest(20, "Installs")

fig, ax = return_plot()

sns.barplot(x="Installs", y="App", data=top_20_installed_apps)



# Popular apps all have more than 10e9 installs and I can't differentiate it because originally it was present in "10000+" format, but I have truncated the '+' and considered it as 10000.

# Most of the apps in this list are preconfigured in the android phones. 
# Let's check the relationship between the number of installs and rating

fig, ax = return_plot()

sns.scatterplot(x="Installs", y="Rating", data=data, ax=ax)
# Top 20 apps depending on the number of reviews

top_20_reviewed_apps = data.nlargest(20, "Reviews")

fig, ax = return_plot()

sns.barplot(x="Reviews", y="App", data=top_20_reviewed_apps)



# Facebook is the most reviewed app in the PlayStore. If we look at the top-3 apps, all of them are from facebook.
# Most popular category with the largest number of installs

# From the plot below, we can find that "Game" category is the most popular category 

category_installs = data.groupby("Category")[["Installs"]].sum().sort_values(by="Installs", ascending=False).reset_index()

fig, ax = return_plot()

sns.barplot(x="Category", y="Installs", data=category_installs, ax=ax)

plt.xticks(rotation=90)
# Variation of rating with category

fig, ax = return_plot(figsize=(20,8))

sns.boxplot(x="Category", y="Rating", data=data, ax=ax)

plt.xticks(rotation=90)
# Free apps and paid apps in each catgory

fig, ax = return_plot()

data.groupby(["Category", "Type"])["App"].count().unstack().fillna(0).plot(kind="bar", stacked=True, ax=ax)

# From the plot, it looks like category "Family" consists most number of "Free" as well as "Paid" apps.
# Different Genres

fig, ax = return_plot(figsize=(16, 20))

sns.countplot(y="Genres", data=data, order=data["Genres"].value_counts().index, ax=ax)
# How the rating varies with the size of the app

fig, ax = return_plot()

sns.scatterplot(x="Size", y="Rating", data=data, ax=ax)
fig, ax = return_plot()

sns.boxplot(x="Content Rating", y="Rating", data=data, ax=ax)
# Let's make a copy of the data for model fitting and drop the App name column from the model_data.

model_data = data.copy()

# Separate the "Rating" column from the data as it will be the dependent variable

model_data_output = model_data["Rating"].copy()
# One hot encoding of "Category", "Content Rating", "Genres", "Subgenres" column and remove the original category column

categories_dummy = pd.get_dummies(data["Category"], prefix="Category", drop_first=True)

type_dummy = pd.get_dummies(data["Type"], prefix="Type", drop_first=True)

content_rating_dummy = pd.get_dummies(data["Content Rating"], prefix="Content", drop_first=True)



model_data = pd.concat([model_data, categories_dummy, type_dummy, content_rating_dummy], axis=1)
model_data.columns
model_data.drop(['Category','Type','Content Rating', 'App', 'Genres', 'Subgenres'], axis=1, inplace=True)
# Data Scaling

scaler = StandardScaler()

model_data_transfromed = scaler.fit_transform(model_data)
# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(model_data_transfromed, model_data_output, 

                                                    test_size=0.25, random_state=42)

print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

print(f"y_train shape: {y_train.shape}")

print(f"y_test shape: {y_test.shape}")
lr = LinearRegression()

lr.fit(X_train, y_train)
print("Cross validation score:", cross_val_score(lr, X_train, y_train, cv=10, scoring='r2').mean() * 100)

print("Linear Regression Score:", lr.score(X_test, y_test))
prediction = lr.predict(X_test)

print("Mean Squared Error:", metrics.mean_squared_error(y_test, prediction))

print("R2 Score:", metrics.r2_score(y_test, prediction))
dtr = RandomForestRegressor(n_estimators=20)

dtr.fit(X_train, y_train)
print("Cross validation score:", cross_val_score(dtr, X_train, y_train, cv=10, scoring='r2').mean() * 100)

print("Random Forest Regression Score:", dtr.score(X_test, y_test))
prediction = dtr.predict(X_test)

print("Mean Squared Error:", metrics.mean_squared_error(y_test, prediction))

print("R2 Score:", metrics.r2_score(y_test, prediction))