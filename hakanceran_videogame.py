# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



# word cloud library

from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
# Video Game Sales Hakkında



print(df.info())
# Data'nın ilk 10 değerlerine bakalım.

df.head(10)
# Sütunların isimleri

print(df.columns)
df.dtypes
df.corr()
df.describe()
print(df.shape)
data1 = df['Platform'].value_counts(dropna = False)

data2 = df['Publisher'].value_counts(dropna = False)

data3 = df['Genre'].value_counts(dropna = False)

data4 = df['Year'].value_counts(dropna = False)



print("\nPlatform",data1.shape,"\nPublisher",data2.shape,"\nGenre",data3.shape,"\nYear",data4.shape,"\n")





# data = pd.concat([data1, data2, data3, data4],axis = 1,ignore_index =True)

# data.head(10)
print(df['Genre'].value_counts(dropna = False))
#correlation map



f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
platGenre = pd.crosstab(df.Platform,df.Platform)



platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)



plt.figure(figsize=(21,10))

sns.barplot(x = platGenreTotal.index, y = platGenreTotal.values, orient='v')



plt.ylabel = "Platform"

plt.xlabel = "Oyun miktarı"

plt.show()
platGenre = pd.crosstab(df.Genre,df.Genre)



platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)



plt.figure(figsize=(21,10))

sns.barplot(x = platGenreTotal.index, y = platGenreTotal.values, orient='v')



plt.ylabel = "Oyun Türü"

plt.xlabel = "Oyun miktarı"

plt.show()
platGenre = pd.crosstab(df.Publisher,df.Publisher)



platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False).head(15)



plt.figure(figsize=(45,10))

sns.barplot(x = platGenreTotal.index, y = platGenreTotal.values, orient='v')



plt.ylabel = "Yapımcı"

plt.xlabel = "Oyun miktarı"

plt.show()
platGenre = pd.crosstab(df.Year,df.Year)



platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)



plt.figure(figsize=(33,10))

sns.barplot(x = platGenreTotal.index, y = platGenreTotal.values, orient='v')



plt.ylabel = "Yıl"

plt.xlabel = "Oyun miktarı"

plt.show()
y1 = df.loc[:,['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum(axis = 0).sort_values(ascending = False)



# visualization

plt.figure(figsize=(13,7))

ax = sns.barplot(x=y1.index, y=y1, palette="deep")

plt.xticks(rotation = 0)

#plt.xlabel("Sales")

#plt.ylabel('Amount')

plt.title('Video Games Sales')
y1 = df.loc[:,['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum(axis = 0).sort_values(ascending = False)



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = y1.index

sizes = y1

explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice



# visualization

plt.figure(figsize = (10,10))

patches, texts, autotexts = plt.pie(

    sizes,

    explode = explode,

    labels = labels,

    autopct = '%1.1f%%',

    shadow = False,

    startangle = 90,

    pctdistance = 0.7,

    #radius = 2.5,

    textprops={'size': 'small'},

    )

plt.title('Video Games Sales', fontsize = 15)

plt.show()
area_list = list(df['Genre'].unique())



NA_Sales = []

EU_Sales = []

JP_Sales = []

Other_Sales = []

Global_Sales = []



for i in area_list:

    x = df[df['Genre'] == i]

    NA_Sales.append(sum(x.NA_Sales) / len(x))

    EU_Sales.append(sum(x.EU_Sales) / len(x))

    JP_Sales.append(sum(x.JP_Sales) / len(x))

    Other_Sales.append(sum(x.Other_Sales) / len(x))

    Global_Sales.append(sum(x.Global_Sales) / len(x))



# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=Global_Sales, y=area_list, color='red', alpha = 0.6, label='Global_Sales')

sns.barplot(x=NA_Sales, y=area_list, color='green', alpha = 0.7, label='NA_Sales' )

sns.barplot(x=EU_Sales, y=area_list, color='blue', alpha = 0.8, label='EU_Sales')

sns.barplot(x=JP_Sales, y=area_list, color='cyan', alpha = 0.9, label='JP_Sales')

sns.barplot(x=Other_Sales, y=area_list, color='yellow', alpha = 1, label='Other_Sales')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Sales', ylabel='Genre',title = "Percentage of Sales to Genres")
# data prepararion



x2011 = df.Name[df.Year == 2011]



plt.subplots(figsize=(18,18))



wordcloud = WordCloud(

    background_color = 'white',

    width = 1280,

    height = 720

).generate(" ".join(x2011))



plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('graph.png')



plt.show()
# data prepararion



x2011 = df.Name



plt.subplots(figsize=(18,18))



wordcloud = WordCloud(

    background_color = 'white',

    width = 1280,

    height = 720

).generate(" ".join(x2011))



plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('graph.png')



plt.show()