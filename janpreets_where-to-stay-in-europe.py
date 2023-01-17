# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Hotel_Reviews.csv")
data.head()
for i in data.columns:

    print(i)
len(data.Hotel_Name.unique()) #total number of distinct hotels
import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 50, 18

rcParams["axes.labelsize"] = 16

from pandas import read_csv

from pandas import datetime

from matplotlib import pyplot

import seaborn as sns
data_plot = data[["Hotel_Name","Average_Score"]].drop_duplicates()

sns.set(font_scale = 2.5)

a4_dims = (30, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.countplot(ax = ax,x = "Average_Score",data=data_plot)
text = ""

for i in range(data.shape[0]):

    text = " ".join([text,data["Reviewer_Nationality"].values[i]])
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=312)

plt.imshow(wordcloud)

plt.title("Wordcloud for countries ")

plt.axis("off")

plt.show()
len(data.Hotel_Name.unique())
data["pos_count"] = 1

data["neg_count"] = 1
data["pos_count"] = data.apply(lambda x: 0 if x["Positive_Review"] == 'No Positive' else x["pos_count"],axis =1)
data["pos_count"].value_counts()
data["neg_count"] = data.apply(lambda x: 0 if x["Negative_Review"] == 'No Negative' else x["neg_count"],axis =1)
data["neg_count"].value_counts()
reviews = pd.DataFrame(data.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())
# reviews.head()

reviews["Hotel_Name"] = reviews.index

reviews.index = range(reviews.shape[0])
reviews.head()
reviews["total"] = reviews["pos_count"] + reviews["neg_count"]
data["count"] = 1

count_review = data.groupby("Hotel_Name",as_index=False)["count"].sum()
reviews = pd.merge(reviews,count_review,on = "Hotel_Name",how = "left")
reviews.head()
for i in reviews.sort_values(by = "count",ascending=False)["Hotel_Name"].head(10).values:

    print(i)
reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")
famous_hotels = reviews.sort_values(by = "count",ascending=False).head(100)
pd.set_option('display.max_colwidth', 2000)

popular = famous_hotels["Hotel_Name"].values[:10]

data.loc[data['Hotel_Name'].isin(popular)][["Hotel_Name","Hotel_Address"]].drop_duplicates()
for i in famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10):

    print(i)
pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["Hotel_Name"].head(10).values

data.loc[data['Hotel_Name'].isin(pos)][["Hotel_Name","Hotel_Address"]].drop_duplicates()
data.Review_Date = pd.to_datetime(data.Review_Date)
temp = data.groupby("Hotel_Name", as_index=False)["Reviewer_Score"].agg([np.mean, np.std]).sort_values("mean",ascending=False)

temp = temp[temp["mean"] > 8.9]

temp.shape

temp.sort_values("std").index[0:20]
lis = ['H10 Casa Mimosa 4 Sup', 'Hotel Casa Camper',

       'H tel de La Tamise Esprit de France', 'Le Narcisse Blanc Spa',

       'Hotel Eiffel Blomet', '45 Park Lane Dorchester Collection', '41',

       'Hotel Stendhal Place Vend me Paris MGallery by Sofitel',

       'H tel D Aubusson', 'Hotel The Serras', 'Hotel Am Stephansplatz',

       'Lansbury Heritage Hotel', 'Covent Garden Hotel', 'The Soho Hotel',

       'Catalonia Magdalenes', 'H tel Saint Paul Rive Gauche',

       'Milestone Hotel Kensington', 'Ritz Paris', 'H tel Fabric',

       'Le 123 S bastopol Astotel']

data.loc[data['Hotel_Name'].isin(lis)][["Hotel_Name","Hotel_Address","Average_Score"]].drop_duplicates()