import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/videogamesales/vgsales.csv")
df.head()
df.columns=["rank","name","platform","year","genre","publisher","na_sales","eu_sales","jp_sales","other_sales","global_sales"]
df.tail()
df.isnull().sum()
def impute_median(series):

    return series.fillna(series.median())
df.year =df["year"].transform(impute_median)
df["year"]=df["year"].apply(lambda x: str(x).replace('.0','') if '' in str(x) else str(x))
print(df["publisher"].mode())
df["publisher"].fillna(str(df["publisher"].mode().values[0]),inplace=True)
df.isnull().sum()
df.shape
df.info()
df.describe().T
fig = plt.figure(figsize = (10,10))

ax = fig.gca()

df.hist(ax = ax)

plt.show()
df.nunique()
plt.figure(figsize=(18,6))

result = df.groupby(["genre"])['global_sales'].aggregate(np.median).reset_index().sort_values('global_sales')

sns.barplot(y=df['global_sales'], x=df["genre"], data=df, order=result['genre'])

plt.title('genre by global_sales')

plt.show()
plt.figure(figsize=(18,6))

result = df.groupby(["platform"])['global_sales'].aggregate(np.median).reset_index().sort_values('global_sales')

sns.barplot(y=df['global_sales'], x=df["platform"], data=df, order=result['platform'])

plt.title('platform by global_sales')

plt.show()
fig=plt.figure(figsize=(18,10))

sns.boxplot(df.genre[:750],df.global_sales[:750])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(18,10))

ax = df.na_sales[:100].plot.kde()

ax = df.eu_sales[:100].plot.kde()

ax = df.jp_sales[:100].plot.kde()

ax = df.other_sales[:100].plot.kde()

ax= df.global_sales[:100].plot.kde()

ax.legend()

plt.show()
labels = df.genre.value_counts().index

colors = ["orange","gray","silver","whitesmoke","rosybrown",

          "firebrick","red","darksalmon","sienna","sandybrown",

          "bisque","tan"]

explode = [0,0,0,0,0,

           0,0,0,0,0,

           0,0]

sizes = df.genre.value_counts().values



# visual 

plt.figure(0,figsize = (18,18))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Games According to Genre',color = 'blue',fontsize = 15)

plt.show()
labels = df.platform.value_counts().index

colors = ["orange","gray","silver","whitesmoke","rosybrown",

          "firebrick","red","darksalmon","sienna","sandybrown",

          "bisque","tan","moccasin","floralwhite","gold",

          "darkkhaki","olivedrab","palegreen","lightseagreen","darkcyan",

          "deepskyblue","lime","tomato","mediumpurple","maroon",

          "coral","olive","yellowgreen","violet","crimson",

          "pink"]

explode = [0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0]

sizes = df.platform.value_counts().values



# visual 

plt.figure(0,figsize = (18,18))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Games According to Platform',color = 'blue',fontsize = 15)

plt.show()
labels = df.year.value_counts().index

colors = ["orange","gray","silver","whitesmoke","rosybrown",

          "firebrick","red","darksalmon","sienna","sandybrown",

          "bisque","tan","moccasin","floralwhite","gold",

          "darkkhaki","olivedrab","palegreen","lightseagreen","darkcyan",

          "deepskyblue","lime","tomato","mediumpurple","maroon",

          "coral","olive","yellowgreen","violet","crimson",

          "pink","hotpink","navajowhite","peachpuff","powderblue",

         "palegoldenrod","mediumturquoise","dodgerblue","royalblue"]

explode = [0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0,0,

           0,0,0,0]

sizes = df.year.value_counts().values



# visual 

plt.figure(0,figsize = (18,18))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Games According to Year',color = 'blue',fontsize = 15)

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.publisher)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(18,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(18,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()