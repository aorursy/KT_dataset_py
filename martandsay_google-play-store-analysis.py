# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# LIKE US ON FB: https://www.facebook.com/codemakerz

# Any results you write to the current directory are saved as output.
# load data

df = pd.read_csv("../input/googleplaystore.csv", encoding='utf-8')
# So we got our dataset

df.head() 

# Here in our dataset we can see Result = 0, 1 that mean 1 mean selected celebrity has cancer and 0 means no cancer
# you can also mention the number of rows

df.head(7)
# so before visualization lets do some EDA. Exploratory data analysis.
# Lets find is there any missing values?

df.isnull().sum() # We can see in Rating we have 1474 missing values & 1 value in Type, 1 in content rating and others also
# Lets see the all columns names

df.columns
# Find the datatype of all the columns 

df.dtypes
# total number of records in dataset

df.count()
# Summary statistics. It will show you the summary statistic for all the numerical values

df.describe()
# info is also used to see some important statistics for dataset.

df.info()
# This will show you top 5 Name columns. As head return top 5

df.App.head()
# you can also do above same thing with array type

df[["App"]].head()
# show multiple columns

df[["App", "Rating"]].head()
# you can also watch below 5 using tail

df[["App", "Rating"]].tail()
# Basic statistics

print("Mean Age: ", df.Rating.mean())

print("Median Age: ", df.Rating.median())

print("Variance Age: ", df.Rating.var())

print("Standard Deviation Age: ", df.Rating.std())

print("25th Percentile of Age: ", df.Rating.quantile(.25))

print("50th Percentile of Age: ", df.Rating.quantile(.5))

print("75th Percentile of Age: ", df.Rating.quantile(.75))
# position based indexing

# We use iloc for this. Here we mention [row, cols] like this.

# : as row means all the rows & 1:4 as columns means 1st index(2nd columns aqs index start from 0), 3rd column and 4th column. 4th index will not included/

df.iloc[:, 1:4].head()
# name based indexing

# in below example we are using all the rows(:) and only Name and Age column.

# you can use any indexing method but i usually prefer iloc. As i am more comfortable with indices.

df.loc[:,["App", "Rating"]].head()
# lets find out different type of Type column

df.Type.value_counts()

# we can see the number of free and paid apps
# Plot Univariate distribution for Ratings using histogram. It will show the total number or app according to the ratings.

# Total number of app acccording to  the ratings

plt.figure(figsize= (15, 10));

#plt.hist(x=df.Rating, color='c');

df.Rating.plot(kind="hist", bins=30, color='c')

plt.title("Univariate: Rating Histogram");

plt.xlabel("Rating");

plt.ylabel("Total Counts");

plt.xlim(right=6)

plt.plot();
# lets see how many categories are there and total number of apps grouped by category

df.Category.value_counts()

# So it clearly show the count of apps for a particular category. value_counts() is very handly function for this.
# but really saying this data is not convincing... we can make it more appealing by plotting it.

# lets plot it.

plt.figure(figsize=(15,10))

df.Category.value_counts().plot(kind="bar", rot=0, title="Pclass Vs Count", color='c', );

plt.xticks(rotation='vertical');

plt.title("Category Vs App Count")

plt.xlabel("Category Name")

plt.ylabel("Count")

plt.plot();
# So now from above example we can see that mostly apps belongs to the family, games , tool category.

# and beauty categroy app is least in number.
# Lets plot the average rating of eah category.

df.groupby(["Category"]).Rating.mean()

# We can see 1.9 category has a rating of 19.0.. which is not possible as our rating is out of 5.

# So definitly it may be some data issue. Now what to do?
df[df["Category"] == '1.9'] # We can see Rating has a value of 19.0, which is not possible.

# So we have two choices:

# 1. Either we remove this row

# 2. Correct this value.

# i will not prefer removing row as it is the only row related to the category 1.9.

# lets replace this value with average rating.
# inplace will save your change to the original dataframe

df["Rating"].replace(19.0, df.Rating.mean(), inplace=True)
# now lets check again

df[df["Category"] == '1.9'] # So we have finally replaced it to the average value
df.groupby(["Category"]).Rating.mean() # So we can see now average values are correct for all categries
plt.figure(figsize=(15,10))

df.groupby(["Category"]).Rating.mean().plot(kind="bar", rot=0, title="Pclass Vs Count");

plt.xticks(rotation='vertical');

plt.xlabel("Category Name")

plt.ylabel("Avg. Rating")

plt.title("Category Vs Avg. Rating")

plt.plot();
# so now we can see the average rating. As per our data average rating is almost same for all so bars are almost equal.
df.groupby(["Category", "Installs"]).Rating.mean()
# Lets find out how many apps are paid or free.

df.Type.value_counts()
plt.figure(figsize=(15, 10));

df.Type.value_counts().plot(kind="bar");

plt.xlabel("License Type");

plt.ylabel("Counts")

plt.title("License Type Vs Counts")

plt.plot();
#So from our above diagrame we can see we have huge number of free apps.
# Lets try to find out the content rating.. that mean how many adult apps are there or how many non-adult apps.

df.columns # it will select content rating column

df['Content Rating'].value_counts() # you can see we have 6 categories in content rating.
plt.figure(figsize=(15, 10), dpi=100, );

df["Content Rating"].value_counts().plot(kind="bar");

plt.title("Content Vs Total Count");

plt.xlabel("Content Type");

plt.ylabel("Count");

plt.ylim(bottom=-10);

plt.plot();
# Actually you can see last two bars are almost hidden because the value is very less.
# now lets see apps grouped by os version 

df["Android Ver"].value_counts()
df[df["Android Ver"] == '1.0 and up'] # we can verify the above data by using this command. You can

# simply replace the value of OS and get the number.
plt.figure(figsize=(15, 10));

plt.xlabel("Anroid Version");

plt.ylabel("Total Apps");

plt.title("Android Version Vs Total Conts");

df["Android Ver"].value_counts().plot(kind="bar");

plt.plot();
df["Android Ver"].value_counts(normalize=True) # You can get the percentage
df["Genres"].value_counts() # here can find all the genres.. Try its plot yourself
# Lets plot top 10 genres

plt.figure(figsize=(15, 10));

df["Genres"].value_counts().head(10).plot(kind="bar");

#df.groupby(["Genres"])..head(10).plot(kind="bar");

plt.title("Top 10 Genres");

plt.xlabel("Genres");

plt.ylabel("Count");

plt.plot();
# you can verify using below statement.

# So found the genre wise installs

df[df["Genres"] == 'Video Players & Editors;Music & Video']
# We need to change the dtype of review but there is one column whih contains 3.0M

df.Reviews.replace("3.0M", '3000000', inplace=True)

df[df.Reviews == "3.0M"]
df.Reviews = df.Reviews.astype("float") # lets change the datatype of Ratings


df_new = df.groupby(["Genres", "Category"], as_index=False).sum()[["Genres", "Category", "Reviews"]].sort_values(by="Reviews", ascending=False).head(10)

df_new
# Below diagram shows the ;top 10 famous genres as per there review counts

plt.figure(figsize=(15, 10));

plt.title("Top 10 Most Reviewed Genres");

plt.xlabel("Genres");

plt.ylabel("Count");

df_new.groupby("Genres").Reviews.sum().sort_values(ascending=False).plot(kind='bar');

plt.plot();
# lets find out category wise application type, how many free and paid apps are there in a category.

pd.crosstab(df["Category"], df["Type"]).apply(lambda r: r/r.sum(), axis=1)
# AS we can see there is one app with type 0, which is obviously not the proper type. Lets fetch that row.

indexNames = df[df["Type"]== '0'].index # find the index

# So we can see its pricing is also not there(everything) and installs section is free. So in this case this data seems to be improper.

# as it doesnt have any exceptional infomrmation so we can remove this.

# Delete these row indexes from dataFrame

df.drop(indexNames , inplace=True) # delete the row
pd.crosstab(df["Category"], df["Type"]).plot(kind="bar",figsize=(15, 10));

plt.plot();

# Handle missing values

df.isnull().sum() # we can see 4 columns have missing values. Lets start with Type
df[df.Type.isnull()] # lets fetch first
# As we can clearly see, its price is 0 that mean it is from free type. So we can replace it.

df.Type.fillna("Free", inplace=True) # replaced
df[df.Type.isnull()]  # No null values
# now lets handle android version column missing values

df[df['Android Ver'].isnull()]
# We can see we have android ver for both of the records. Lets see a crosstab for android ver and current ver

pd.crosstab(df[(df['Current Ver'] == '4.4')]['Current Ver'], df[(df['Current Ver'] == '4.4')]["Android Ver"])
pd.crosstab(df[(df['Current Ver'] == '4.4')]['Current Ver'], df['Android Ver']).plot(kind='bar', figsize=(15, 10))
# We can see there is last updated column, we canuse it to extact year and then we can find out which Andriod Version was popular at that time.

# Seems like a good finding.

# Lets create one function to extract year from the Last Updated.

def get_year(date):

    year = date.split(',')[1]

    year = year.strip()

    return year
df["Last Updated"].map(lambda x: get_year(x)) # we get the year
# lets create a new column for year

df["Year"] = df["Last Updated"].map(lambda x: get_year(x))
df.head()
# now lets plot crosstab for year and android ver

pd.crosstab(df[df["Year"] == '2018'].Year, df["Android Ver"])
pd.crosstab(df[df["Year"] == '2018'].Year, df["Android Ver"]).plot(kind='bar', figsize=(15, 10))
# so we can see in year 2018 mostly app were using android version 4.1 and up. So this seems a fair analysis. Let replace with this value.

most_frequent_andoid = df['Android Ver'].value_counts().idxmax()

df["Android Ver"].fillna(most_frequent_andoid, inplace=True)
# lets check again for null values in android version

df[df["Android Ver"].isnull()] # So no records... its done.
# lets check again for null values in whole data frame

df.isnull().sum() # so we can see rating has a big number of missing values. Lets deal with it.
pd.options.display.max_rows  = 15  # it will display always 15 rows.
# lets fetch all the missing value rows for Rating

df[df["Rating"].isnull()] # there are total 1474 rows
 #Lets get the median review according to genre and category

df.groupby(["Genres", "Category"]).Rating.median()
df.groupby(["Genres", "Category"]).Rating.median().plot(kind="bar", figsize=(30, 10));

# so we can see Ratings for all the genres and category are simillar. 
# lets find out the median rating of total df

df.Rating.median() # in this case our median is also simillar to category and genres wise grouped data. so lets replace with this data only.
median_rating = df.Rating.median()

df.Rating.fillna(median_rating, inplace = True)
df[df.Rating.isnull()] # so finally we replaced all the missing values. Lets check it
df.isnull().sum() # so finally we got rid of missing values.