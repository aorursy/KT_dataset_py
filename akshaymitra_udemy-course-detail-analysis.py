import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
df=pd.read_csv("../input/udemy-courses/udemy_courses.csv")
df.head()
df.columns
df.shape
print("Percentage of paid courses is: ",df[df["is_paid"] == True].shape[0]*100/df.shape[0])

print("Percentage of free courses is: ",100 - df[df["is_paid"] == True].shape[0]*100/df.shape[0])

df.subject.unique()
df.level.unique()
sns.catplot(x="level", col="is_paid",

                data=df, kind="count",

                height=8, aspect=.9);
fig_dims = (9, 6)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x="level",y="num_subscribers",data=df,ax=ax,palette="cubehelix")
df["price"]=df.price.replace("Free",0)

df["price"]=df.price.replace("TRUE","178")
fig_dims = (6,9)

fig, ax = plt.subplots(figsize=fig_dims)

df['price'] = df['price'].astype(int)

plt.hist(df["price"])
fig_dims = (15, 6)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x="subject",y="num_subscribers",data=df,ax=ax,palette="cubehelix",estimator=np.mean,hue=df.level)
sns.catplot(x="subject",y="price",hue="level",data=df,kind="box",height=9, aspect=2.3)
text_webdevelopment = " ".join(review for review in df[df["subject"] =="Web Development"]["course_title"])

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_webdevelopment)

plt.figure(figsize = (10, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text_busfin = " ".join(review for review in df[df["subject"] =="Business Finance"]["course_title"])

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_busfin)

plt.figure(figsize = (10, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text_graphdes = " ".join(review for review in df[df["subject"] =="Graphic Design"]["course_title"])

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_graphdes)

plt.figure(figsize = (10, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text_musinstru = " ".join(review for review in df[df["subject"] =="Musical Instruments"]["course_title"])

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(text_musinstru)

plt.figure(figsize = (10, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()