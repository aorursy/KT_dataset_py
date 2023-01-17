# Importing Libraries for Loading Dataset

import numpy as np

import pandas as pd
# Importing required libraries for data visualisation

import matplotlib.pyplot as plt

import seaborn as sns
objects = pd.read_csv("../input/startup-investments/objects.csv", low_memory=False)

objects.head()
# Rename id in objects.csv to founded_object_id

objects.rename(columns={'id':'funded_object_id'}, inplace=True)

objects.head()
objects.info()
objects.drop(["created_at","updated_at", "logo_url", "logo_width","overview","category_code","status", "permalink", "entity_id","parent_id","normalized_name", "logo_height","short_description", "created_at", "updated_at", "twitter_username","relationships", "domain", "homepage_url", "overview", "tag_list", "country_code","city", "region", "state_code"], axis="columns", inplace=True)

objects.info()
investments = pd.read_csv("../input/startup-investments/investments.csv")

investments.head()
investments.funded_object_id
# Loading and merging the required dataset



df = investments.merge(objects, on='funded_object_id')

df.head()
df.info()
df.drop(["closed_at", "first_investment_at","invested_companies", "investment_rounds", "created_at", "updated_at"], axis="columns", inplace= True)
df.head()
# Using a heatmap to check the missing data

plt.figure(figsize=(10,7))

sns.heatmap(df.isnull(), yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='funded_object_id', data=df)