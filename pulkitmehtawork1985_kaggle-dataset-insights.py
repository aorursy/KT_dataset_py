# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.set_option('display.max_colwidth', -1)

# Load data



dataset_df = pd.read_csv("../input/meta-kaggle/Datasets.csv")

dataset_votes = pd.read_csv('../input/meta-kaggle/DatasetVotes.csv')

datasources = pd.read_csv('../input/meta-kaggle/Datasources.csv')

datasources_object = pd.read_csv('../input/meta-kaggle/DatasourceObjects.csv')

user = pd.read_csv('../input/meta-kaggle/Users.csv')
# lets see total number of datasets.

dataset_df.shape
# explore few rows

dataset_df.head(3)
# Let's check average of total votes per dataset.

dataset_df['TotalVotes'].mean()
dataset_df['TotalKernels'].sum() , dataset_df['TotalKernels'].mean() 
# Let's find dataset with max votes.



dataset_df[dataset_df['TotalVotes'] == dataset_df['TotalVotes'].max()]
# lets check which user has most upvotes in a dataset

user[user['Id'] == 14069]
# Lets check which dataset got max votes

datasources_object[datasources_object.DatasourceVersionId == 23502.0]
# Let's check which tags have most datasets

tags = pd.read_csv("../input/meta-kaggle/Tags.csv")
tags.groupby('Name')['DatasetCount'].sum().sort_values(ascending = False).head(10)
top_10_tags = tags.groupby('Name')['DatasetCount'].sum().sort_values(ascending = False).head(10).reset_index()

# Bar chart using matplotlib



plt.figure(figsize = (20,10))

plt.bar(top_10_tags['Name'],top_10_tags['DatasetCount'])

plt.xlabel('Tag Name',fontsize = 15)

plt.ylabel('Dataset Counts',fontsize = 15)

plt.xticks(top_10_tags['Name'],  fontsize=10, rotation=30)

plt.title(' Dataset Tags Popularity',fontsize = 30)

plt.show()

### How many users have uploaded dataset

dataset_df['CreatorUserId'].nunique()
# Let's explore Kaggle dataset year by year



# change creation date to datetime

dataset_df['CreationDate'] = pd.to_datetime(dataset_df['CreationDate'] )
# extract year

dataset_df['Year'] = dataset_df['CreationDate'].dt.year
# Year on year count of datasets . It is expected to increase dramatically.

dataset_df.groupby('Year')["Id"].count()
year_ds = dataset_df.groupby('Year')["Id"].count().reset_index()

plt.plot(year_ds['Year'],year_ds['Id'])
user.head()