import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import itertools
work_dir = "../input/google-play-store-apps/"
app_data = pd.read_csv(os.path.join(work_dir, 'googleplaystore.csv'))
app_data.head()
app_data.info()
app_data.isnull().sum()
per_missing = (app_data.isnull().sum()/app_data.count())*100
per_missing
sns.heatmap(app_data.isnull(), cbar=False, yticklabels=False);
plt.figure(figsize=(20,8))

sns.set_style('whitegrid')

sns.boxenplot(x='Rating', data=app_data);
plt.figure(figsize=(20,8))

plt.xlim(0,6)

sns.set_style('whitegrid')

sns.boxenplot(x='Rating', data=app_data);
app_data.columns
app_data['Category'].unique()
app_data.groupby(by='Category').mean()
app_data.Rating.mean()
app_data.groupby(by='Category').mean().index[1:]
app_data.groupby(by='Category').mean().values[1:]
averages = dict(zip(app_data.groupby(by='Category').mean().index[1:], app_data.groupby(by='Category').mean().values.tolist()[1:]))
averages['BUSINESS']
df_small = app_data.iloc[:500]
def category_fill(x):

    rating = x[0]

    category = x[1]

    if pd.isnull(rating):

        if category in averages.keys():

            return averages[category][0]



    else:

        return rating

app_data['new_rating'] = app_data[['Rating', 'Category']].apply(category_fill, axis=1)
plt.ylim(0,6)

sns.boxplot(data=app_data['new_rating']);
plt.ylim(0,6)

sns.boxplot(data=app_data['Rating']);
app_data.drop('Rating', axis=1, inplace=True)
app_data.dropna(inplace=True)
app_data.info()
app_data['Reviews'] = app_data.Reviews.astype(int)
app_data.Size.str.findall(r'\d')
def merge_size(size):

    

    if len(size)!=0:

        return ''.join([str(i) for i in size])

    else:

        return 0
app_data["Size"] = app_data.Size.str.findall(r'\d').apply(merge_size).astype(int)
app_data["Installs"] = app_data.Installs.str.replace('+','').str.replace(',','').astype(int)
app_data.Type.unique()
app_data["Price"] = app_data.Price.str.replace('$', '').astype(float)
app_data["Content Rating"].unique()
app_data.Genres.unique()
app_data["Last Updated"] = pd.to_datetime(app_data["Last Updated"])
app_data["Current Ver"].describe()
app_data["Android Ver"].unique()
app_data["Android Ver"].str.split(' ')
def get_ver(string):

    data = string.split(" ")

    return data[0]
df_small['Android Ver'].apply(get_ver).unique()
app_data["Android Ver"] = app_data['Android Ver'].apply(get_ver)
app_data["Current Ver"] = app_data['Current Ver'].apply(get_ver)
app_data.info()
plt.figure(figsize=(20,10))

plt.xlabel("Number of apps", fontsize=30)

plt.ylabel("Category of apps", fontsize=30)

plt.title("How category of apps are distributed?", fontsize=30)

ax = app_data['Category'].value_counts().plot(kind='barh', fontsize=15);

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Number of apps", fontsize=20)

plt.ylabel("Genres of apps", fontsize=20)

plt.title("How genres of apps are distributed?", fontsize=30)

ax = app_data.Genres.value_counts().plot(kind='barh', fontsize=15);

ax.xaxis.label.set_size(30)

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Number of apps", fontsize=20)

plt.ylabel("Genres of apps", fontsize=20)

plt.title("How genres of apps are distributed?", fontsize=30)

temp = app_data.Genres.value_counts()

ax = temp[temp.values > 60].plot(kind='barh', fontsize=15);

ax.xaxis.label.set_size(30)

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Price of app in $", fontsize=20)

plt.ylabel("Rating of apps", fontsize=20)

plt.title("Relationship between App Pricing and Rating", fontsize=30)

app_data.groupby(by='Price').mean()['new_rating'].plot(kind='line', fontsize=15);
plt.figure(figsize=(20,10))

plt.xlabel("Number of apps", fontsize=20)

plt.ylabel("Android Verison", fontsize=20)

plt.title("Popular android verison among developers", fontsize=30)

ax = app_data.groupby('Android Ver').count()['Installs'].plot(kind='barh', fontsize=15);

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Number of Installs", fontsize=20)

plt.ylabel("Size of app", fontsize=20)

plt.title("First 100 hunderd installs vs Size of app", fontsize=30)

ax = app_data.groupby('Installs').mean()['Size'].plot(xlim=(0,100), fontsize=15);

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Number of Installs", fontsize=20)

plt.ylabel("Size of app", fontsize=20)

plt.title("Installs vs Size of app", fontsize=30)

ax = app_data.groupby('Installs').mean()['Size'].plot(fontsize=15);

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Number of Installs", fontsize=20)

plt.ylabel("Content Rated", fontsize=20)

plt.title("Content Rating vs Installations", fontsize=30)

ax = app_data.groupby("Content Rating").count()['Installs'].plot(kind='barh',fontsize=15);

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)
plt.figure(figsize=(20,10))

plt.xlabel("Year", fontsize=20)

plt.ylabel("Frequency", fontsize=20)

plt.title("Frequency of Updates", fontsize=30)

ax = app_data['Last Updated'].value_counts().plot(fontsize=15);

ax.yaxis.label.set_size(30)

right_side = ax.spines["right"]

top_side = ax.spines["top"]

right_side.set_visible(False)

top_side.set_visible(False)