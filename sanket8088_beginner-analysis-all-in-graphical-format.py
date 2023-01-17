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

# Any results you write to the current directory are saved as output.


#importing data
import bq_helper
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names.csv")
agg_names.head()
agg_names.shape
#gender wise population 
agg_names.groupby("gender")["name"].count().plot(kind="bar")
#yearwise population growth
agg_names.groupby("year")["name"].count().plot()
#Top 5 male names
agg_names.groupby("gender")["name"].value_counts()["M"].head().plot(kind="bar")
#Top 5 female names
agg_names.groupby("gender")["name"].value_counts()["F"].head().plot(kind="bar")
#Growth of male population
agg_names_male=agg_names[agg_names.gender=="M"]
agg_names_male.groupby("year").name.count().plot()
#Growth of female population
agg_names_female=agg_names[agg_names.gender=="F"]
agg_names_female.groupby("year").name.count().plot()
#last five year gender distribution population
agg_names_2011=agg_names[agg_names.year>=2011]
agg_names_2011.groupby(["year","gender"])["name"].count().plot(kind="bar")
from wordcloud import WordCloud
names_list=agg_names.name.unique().tolist()
sv=WordCloud().generate(" ".join(names_list))
plt.figure(figsize=(10,8))
plt.imshow(sv)