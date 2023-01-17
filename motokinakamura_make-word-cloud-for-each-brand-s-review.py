#　libralies

import pandas as pd

import numpy as np

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#For making word_cloud

from wordcloud import WordCloud
df = pd.read_csv("../input/japanese_whisky_review.csv")
df.head()
df.columns
#change name of columns

df = df.rename(columns={

    "Unnamed: 0":"no.",

    "Bottle_name":"name",

    'Brand':"brand",

    'Title':"title",

    'Review_Content':"review"

})
# see unique values

print("brand")

print(df.brand.unique())

print("")

print("name")

print(df.name.unique())

print("")
sns.countplot(x=df.brand,data=df, palette="Pastel1")
text = df.review
wordcloud = WordCloud(

    background_color="white",

    width=900,

    height=500

).generate(str(text))



plt.figure(figsize=(12,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

print("ALL_reviews")
df.brand.unique()
yamazaki = df[df.brand == "Yamazaki"]

hibiki = df[df.brand == "Hibiki"]

hakushu = df[df.brand == "Hakushu"]

nikka = df[df.brand == "Nikka"]
yamazaki_text = str(yamazaki.review)

hibiki_text = str(hibiki.review)

hakushu_text = str(hakushu.review)

nikka_text = str(nikka.review)
yamazaki_wordcloud = WordCloud(

        background_color="white",

        width=900,

        height=500

    ).generate(yamazaki_text)



hibiki_wordcloud = WordCloud(

        background_color="white",

        width=900,

        height=500

    ).generate(hibiki_text)



hakushu_wordcloud = WordCloud(

        background_color="white",

        width=900,

        height=500

    ).generate(hakushu_text)



nikka_wordcloud = WordCloud(

        background_color="white",

        width=900,

        height=500

    ).generate(nikka_text)


plt.figure(figsize=(12,10))

plt.imshow(yamazaki_wordcloud)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Yamazaki")



plt.figure(figsize=(12,10))

plt.imshow(hibiki_wordcloud)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Hibiki")



plt.figure(figsize=(12,10))

plt.imshow(hakushu_wordcloud)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Hakushu")



plt.figure(figsize=(12,10))

plt.imshow(nikka_wordcloud)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Nikka")
#多くですぎる言葉を出ないようにしてみる

stop_words = [

    "whisky","Yamazaki","Hibiki","Hakushu","Nikka","the","thi","is","and","it","to","of","my","for","this","This","Whiskey","with","not",

    "very","but","as","in","on"]
yamazaki_wordcloud_2 = WordCloud(

        background_color="white",

        width=900,

        height=500,

    stopwords=set(stop_words)

    ).generate(yamazaki_text)



hibiki_wordcloud_2 = WordCloud(

        background_color="white",

        width=900,

        height=500,

        stopwords=set(stop_words)

    ).generate(hibiki_text)



hakushu_wordcloud_2 = WordCloud(

        background_color="white",

        width=900,

        height=500,

        stopwords=set(stop_words)

    ).generate(hakushu_text)



nikka_wordcloud_2 = WordCloud(

        background_color="white",

        width=900,

        height=500,

        stopwords=set(stop_words)

    ).generate(nikka_text)
plt.figure(figsize=(12,10))

plt.imshow(yamazaki_wordcloud_2)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Yamazaki")



plt.figure(figsize=(12,10))

plt.imshow(hibiki_wordcloud_2)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Hibiki")



plt.figure(figsize=(12,10))

plt.imshow(hakushu_wordcloud_2)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Hakushu")



plt.figure(figsize=(12,10))

plt.imshow(nikka_wordcloud_2)

#グリッド線の有無(on/off)

plt.axis("off")

plt.show()

print("Nikka")