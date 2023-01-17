# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/timesData.csv')
data.info()
# convert object to numeric
data["international"] = data["international"].convert_objects(convert_numeric=True)
data["income"] = data["income"].convert_objects(convert_numeric=True)
data["total_score"] = data["total_score"].convert_objects(convert_numeric=True)
data["international_students"] = data["international_students"].convert_objects(convert_numeric=True)
data.dtypes
data.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt=".1f",ax=ax)
plt.show()
data.head()
data.tail()
data.columns
# square of student_staff_ratio
# add new column and use apply method
data["ssr**2"] = data.student_staff_ratio.apply(lambda n : n**2)
data.head()
# we can see our new column ssr**2
data.columns
# citations = university score for citations (research influence)
# income = university score for industry income (knowledge transfer)
data.citations.plot(kind="line",linewidth=1.5,grid=True,alpha=0.5,color="red",linestyle=":",label="citations",
                    figsize=(10,10))
data.income.plot(linewidth=1.5,grid=True,alpha=0.5,color="black",linestyle="-.",label="income",figsize=(10,10))
plt.xlabel("University")
plt.ylabel("Score")
plt.title("University Score for Citations and Industry Income")
plt.legend(loc = "upper right")
plt.show()
# correlation of teaching and research
# research = university score for research
# teaching = university score for teaching
data.plot(kind="scatter", x="research", y="teaching",color="red",alpha=0.5,figsize=(10,10))
plt.xlabel("Research")
plt.ylabel("Teaching")
plt.title("Correlation of Teaching and Research")
plt.show()
# total_score hist
# total_score = total score for university, used to determine rank
fig,ax = plt.subplots(nrows=2,ncols=1)
data.total_score.plot(kind="hist",bins=50,color="purple",figsize=(10,10),normed=True,label="total score",ax=ax[0])
data.total_score.plot(kind="hist",bins=50,color="purple",figsize=(10,10),normed=True,label="total score",cumulative=True,
                     ax=ax[1])
plt.legend()
plt.show()
#sorting
country_name = list(data.country.unique())
teaching_ratio = []
for i in country_name:
    x = data[data.country == i]
    y = sum(x.teaching)/len(x)
    teaching_ratio.append(y)
n_data = pd.DataFrame({"Country Name":country_name,"Teaching Ratio":teaching_ratio})
new_index = (n_data["Teaching Ratio"].sort_values(ascending=False)).index.values
sorted_data = n_data.reindex(new_index)
#visualization
plt.figure(figsize=(20,20))
sns.barplot(x=sorted_data["Country Name"].head(25), y=sorted_data["Teaching Ratio"].head(25))
plt.xticks(rotation=80)
plt.xlabel("Country Name")
plt.ylabel("Teaching Ratio")
plt.title("Teaching Ratio of Countries")
plt.show()
# University numbers of countries in that list
name_count = Counter(data.country)
most_common_countries = name_count.most_common(20)
x,y = zip(*most_common_countries)
x,y = list(x),list(y)
#visualization
plt.figure(figsize=(20,20))
sns.barplot(x=x, y=y, palette=sns.cubehelix_palette(len(x)))
plt.xticks(rotation=45)
plt.xlabel("Country Name")
plt.ylabel("University Number")
plt.title("University Numbers of Countries ")
plt.show()
# filtering data
usa     = data[data.country == "United States of America"]
uk      = data[data.country == "United Kingdom"]
gm      = data[data.country == "Germany"]
au      = data[data.country == "Australia"]                                 
# concatenating data
conc_data = pd.concat([usa,uk,gm,au],axis=0,ignore_index=True)
conc_data
# EDA
conc_data.boxplot(column="teaching", by="country",figsize=(9,9))
plt.show()
# There are a lot of outlier in uk's teaching eda
# USA Scores
usa.plot(subplots=True,figsize=(9,9))
plt.show()
turkey = data[data.country == "Turkey"]
turkey.head()
# tidy data
melted = pd.melt(frame=turkey,id_vars="university_name",value_vars=["teaching","international"])
t = pd.concat([melted.head(),melted.tail()],axis=0)
t
# Missing data
data.info()
data["total_score"].value_counts(dropna=False).head()
# 1402 NaN values in total score
# drop nan values
data1 = data.copy()
data1["total_score"].dropna(inplace=True)
assert data1["total_score"].notnull().all()
data1["total_score"].fillna("empty",inplace=True)
# as you can see we drop all nan values
data1["total_score"].value_counts(dropna=False).head()
# STATISTICAL EXPLORATORY DATA ANALYSIS
data.describe()
# time series
# random times
times = ["1958-11-06","1958-01-19","1959-07-08","1959-01-01","1960-09-12"]
dt = pd.to_datetime(times)
type(dt)
# close warning
import warnings
warnings.filterwarnings("ignore")
#
data2 = data.head()
data2["date"] = dt
data2 = data2.set_index("date")
data2.head()
# ("A") represents year
data2.resample("A").mean()
# ("M") represents mounth
data2.resample("M").mean().head(15)
# acording to mean we change the nan values
data2.resample("M").mean().interpolate("linear").head(15)
# change index
data3 = data.head()
data3 = data3.set_index("world_rank")
data3.head()
data3.index.name = "rank"
data3[["country","university_name","teaching"]].head()