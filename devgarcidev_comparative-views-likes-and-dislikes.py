import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import json



%matplotlib inline
## get categories id and how many times to appear and sorted

data = pd.read_csv('../input/youtube-new/USvideos.csv')

data.groupby('category_id').category_id.count().sort_values(ascending=False)
# Get categories id with his literal

id_to_category = {}



with open("../input/youtube-new/US_category_id.json","r") as f:

    id_data = json.load(f)

    for category in id_data["items"]:

        id_to_category[category["id"]] = category["snippet"]["title"]

id_to_category
# Convert category_id into a string and insert values with literal category

type(data["category_id"][0])

data["category_id"] = data["category_id"].astype(str)

data.insert(4, "category",data["category_id"].map(id_to_category))

display(data)
# How many videos by category

data.groupby('category').category.count().sort_values(ascending=False)
## get how many likes, dislikes by category

#data.groupby('category').likes.sum().sort_values(ascending=False)

#data.groupby('category').dislikes.sum().sort_values(ascending=False)



views_dis_likes = data.groupby(['category'])["views","likes", "dislikes"].apply(lambda x : x.astype(int).sum()).sort_values(by='views', ascending = False)
corr_list = ['views','likes','dislikes']

correlation_data = views_dis_likes[corr_list].corr() 

display(correlation_data)



sns.heatmap(correlation_data,cmap="Blues",annot=True)