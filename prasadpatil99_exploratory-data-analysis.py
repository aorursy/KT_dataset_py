# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.drop(['name','id','host_name','last_review','host_id','calculated_host_listings_count'], axis=1, inplace=True)
df.head()
df.isnull().sum()
df.reviews_per_month.fillna(df.mean(), inplace=True)
img=np.array(Image.open('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.ioff()
plt.show()
ng = " ".join(str(each) for each in df.neighbourhood_group)
nh= " ".join(str(each) for each in df.neighbourhood)
data = (" ".join([ng, nh]))
data = str(data)
from wordcloud import WordCloud, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

text = " ".join(str(each) for each in df.neighbourhood_group)
text = df.neighbourhood_group
wordcloud = WordCloud(max_words=100, stopwords=STOPWORDS, collocations=False).generate(data)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
import folium
import folium.plugins

map = folium.Map([40.7128,-74.0060],zoom_start=11)
folium.plugins.HeatMap(df[['latitude','longitude']].dropna(), radius=8, 
                       gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'black'}).add_to(map)
display(map)
title = 'Locating Neighbourhood Group Location'
plt.figure(figsize=(10,7))
sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group).set_title(title)
plt.ioff()
title = 'Room Type preference as per Location Coordinates'
plt.figure(figsize=(10,7))
sns.scatterplot(df.longitude,df.latitude,hue=df.room_type).set_title(title)
plt.ioff()
sns.catplot(x="room_type", y="price", col="neighbourhood_group", col_wrap=3, data=df);
sns.countplot(x ='room_type', data = df) 
sns.relplot(x="availability_365", y="price", data=df);
sns.catplot(x="availability_365", y="price", col="room_type",
            data=df.query("availability_365 == 0 or availability_365 == 365 "));
sns.relplot(x="minimum_nights", y="price", data=df);
sns.catplot(x="neighbourhood_group", y="price", col="room_type", col_wrap=3,data=df);
df1 = df[df.room_type == "Private room"][["neighbourhood_group","price"]]
sns.barplot(x="neighbourhood_group", y="price", data=df1)
plt.show() 
title = 'Room type taste as per Neighbourhood Group'
sns.catplot(x='room_type', kind="count", hue="neighbourhood_group", data=df);
plt.title(title)
plt.ioff()