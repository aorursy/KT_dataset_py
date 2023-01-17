import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

train_df.head()
train_df.describe()
labels = []

values = []

for col in train_df.columns:

    labels.append(col)

    values.append(train_df[col].isnull().sum())

    print(col, values[-1])
cnt_author = train_df['author'].value_counts()

plt.figure(figsize=(15,10))

sns.barplot(cnt_author.index, cnt_author.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Author', fontsize=12)

plt.title('Count of occurence of author works', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
text_eap = train_df[train_df["author"] == "EAP"]["text"].str.cat(sep = " ")

wordcloud_eap = WordCloud().generate(text_eap)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud_eap, interpolation = "bilinear")

plt.title("Wordcloud : EAP", fontsize=30)

plt.axis("off")
text_eap = train_df[train_df["author"] == "MWS"]["text"].str.cat(sep = " ")

wordcloud_eap = WordCloud().generate(text_eap)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud_eap, interpolation = "bilinear")

plt.title("Wordcloud : MWS", fontsize=30)

plt.axis("off")
text_eap = train_df[train_df["author"] == "HPL"]["text"].str.cat(sep = " ")

wordcloud_eap = WordCloud().generate(text_eap)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud_eap, interpolation = "bilinear")

plt.title("Wordcloud : HPL", fontsize=30)

plt.axis("off")