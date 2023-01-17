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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import nltk

from nltk.corpus import stopwords

set(stopwords.words('english'))

import os

import re

import time

from wordcloud import WordCloud, STOPWORDS 

import warnings

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")

%matplotlib inline
# Reading the provided image

plt.figure(figsize= (5,18))

img_arr= cv2.imread("../input/new-york-city-airbnb-open-data/New_York_City_.png")

plt.title("New York Airbnb setup Location !")

plt.imshow(img_arr)
data= pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()
data.shape
sns.countplot(data["room_type"])
sns.countplot(data["neighbourhood_group"], hue= data["room_type"])
plt.figure(figsize= (12,6))

plt.title("Distribution of the price range of the hotels")

sns.distplot(data["price"])
plt.figure(figsize=(12, 6))

sns.countplot(data[data["minimum_nights"]<50]["minimum_nights"])

plt.xticks(rotation ="vertical")

plt.plot()
plt.figure(figsize=(12, 6))

sns.countplot(data[data["minimum_nights"]>50]["minimum_nights"])

plt.xticks(rotation ="vertical")

plt.plot()
# Plot to visualize the days for the room avaiability

sns.distplot(data["availability_365"])

plt.title('Plot to visualize the days for the room avaiability')

plt.plot()
sns.countplot(data[data["calculated_host_listings_count"]<10]["calculated_host_listings_count"], hue=data["room_type"])
n1_data= data[data["price"]<500]

plt.figure(figsize= (12,6))

sns.violinplot(n1_data["neighbourhood_group"], n1_data["price"])
n1_data.plot(kind="scatter", x="longitude", y="latitude", label= "availability", c="price", cmap= plt.get_cmap('jet'), colorbar= True, figsize= (10,8))
import re 

corpus=[]

for i in range(len(data)):

    nme=re.sub("[^a-zA-Z]"," ",str(data["name"][i]))

    nme=nme.split()

    nme = [word.lower() for word in nme if not word in stopwords.words("english")]

    corpus.extend(nme)
#corpus
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data["name"]))
print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)