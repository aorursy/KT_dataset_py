# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/top-270-rated-computer-science-programing-books/prog_book.csv")

data.head()
data.sort_values(by="Rating",ascending=False)[0:20]
data["Reviews"]=data["Reviews"].str.replace(",",".").astype(float)
data_more_than_50_reviews=data[data["Reviews"]>50]

data_more_than_50_reviews.sort_values(by="Rating",ascending=False)[0:20]
data_more_than_50_reviews.sort_values(by="Rating")[0:20]
data_more_than_50_reviews["Price/Rating"]=data_more_than_50_reviews["Price"]/data_more_than_50_reviews["Rating"]

data_more_than_50_reviews.sort_values(by="Price/Rating")[0:20]
data_more_than_50_reviews[["Price","Price/Rating"]].corr()
from sklearn.preprocessing import MinMaxScaler

data_more_than_50_reviews["Price/Rating_scaled"]=MinMaxScaler().fit_transform(data_more_than_50_reviews["Price"].values.reshape(-1, 1))/(MinMaxScaler().fit_transform

(data_more_than_50_reviews["Rating"].values.reshape(-1, 1)))



data_more_than_50_reviews.sort_values(by="Price/Rating_scaled")[0:20]
data_more_than_50_reviews[["Price","Price/Rating_scaled"]].corr()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

sns.scatterplot(x="Rating",y="Price",data=data_more_than_50_reviews,)
import plotly.express as px

fig = px.scatter(data_more_than_50_reviews, x="Rating", y="Price",hover_name='Book_title')

fig.show()
import plotly.express as px

fig = px.scatter(data_more_than_50_reviews, x="Rating", y="Price",hover_name='Book_title',size="Number_Of_Pages")

fig.show()
from wordcloud import WordCloud, STOPWORDS

wocl=WordCloud(stopwords=STOPWORDS).generate(" ".join(data["Description"].tolist()))

plt.figure(figsize=(15,10))

plt.imshow(wocl)
from wordcloud import WordCloud, STOPWORDS

wocl=WordCloud(stopwords=STOPWORDS).generate(" ".join(data["Book_title"].tolist()))

plt.figure(figsize=(15,10))

plt.imshow(wocl)