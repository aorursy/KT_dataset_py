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
data=pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv",encoding = "ISO-8859-1")

data.head()
turkey=data[data["country_txt"]=="Turkey"]

turkey.head()
turkey.tail()
len(turkey)
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))

sns.countplot(data=turkey,x="imonth")
scite=turkey["scite1"]

text=""

for cite in scite:

    text=text+str(cite)+" "


from wordcloud import STOPWORDS

STOPWORDS.add("one")

STOPWORDS.add("want")

STOPWORDS.add("didn")

STOPWORDS.add("lot")

STOPWORDS.add("don")

STOPWORDS.add("think")

STOPWORDS.add("anything")

STOPWORDS.add("someone")

STOPWORDS.add("know")

STOPWORDS.add("nan")

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=40,stopwords=STOPWORDS).generate(text.lower())

plt.figure(figsize=(20,40))

plt.imshow(wordcloud)
scite.value_counts().sort_values(ascending=False)[0:20]
plt.figure(figsize=(20,14))

sns.countplot(data=turkey,x="iyear")
s70s=turkey[(turkey["iyear"]>=1970) & (turkey["iyear"]<1980)]

s70s.head()
set(s70s["scite1"])
set(s70s["addnotes"])
plt.figure(figsize=(20,14))

sns.countplot(data=data,x="country_txt")
count=data.groupby("country_txt")["country_txt"].count()

count.sort_values(ascending=False)[0:10]
count.sort_values(ascending=False)[10:20]
twoandlater=data[data["iyear"]>=2000]

count2=twoandlater.groupby("country_txt")["country_txt"].count()

count2.sort_values(ascending=False)[0:10]
count2.sort_values(ascending=False)[10:20]
twoandbefore=data[data["iyear"]<=2000]

count2=twoandbefore.groupby("country_txt")["country_txt"].count()

count2.sort_values(ascending=False)[0:10]
count2.sort_values(ascending=False)[10:20]
twoandbefore=data[data["iyear"]<=2010]

count2=twoandbefore.groupby("country_txt")["country_txt"].count()

count2.sort_values(ascending=False)[0:10]
count2.sort_values(ascending=False)[10:20]