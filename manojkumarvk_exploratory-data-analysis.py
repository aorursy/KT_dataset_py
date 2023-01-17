import numpy as np

import pandas as pd 

import string



import nltk

from nltk.corpus import stopwords



import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt
df = pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")

df.head()
df.describe()
df.Score.value_counts()
sns.countplot(x='Score', data=df, palette='RdBu')

plt.xlabel('Score (Rating)')

plt.show()
temp_df = df[['UserId','HelpfulnessNumerator','HelpfulnessDenominator', 'Summary', 'Text','Score']].copy()

temp_df["Sentiment"] = temp_df["Score"].apply(lambda score: "positive" if score > 3 else ("negative" if score < 3 else "not defined"))

temp_df["Usefulness"] = (temp_df["HelpfulnessNumerator"] / temp_df["HelpfulnessDenominator"]).apply\

(lambda n: ">75%" if n > 0.75 else ("<25%" if n < 0.25 else ("25-75%" if n >= 0.25 and n <= 0.75 else "useless")))

temp_df.loc[temp_df.HelpfulnessDenominator == 0, 'Usefulness'] = "useless"

temp_df.head()
sns.countplot(x='Sentiment', order=["positive", "negative"], data=temp_df, palette='RdBu')

plt.xlabel('Sentiment')

plt.show()
pos = temp_df.loc[temp_df['Sentiment'] == 'positive']

pos = pos[:1000]

neg = temp_df.loc[temp_df['Sentiment'] == 'negative']

neg = neg[:1000]
def create_Word_Corpus(temp):

    words_corpus = ''

    for val in temp["Summary"]:

        text = str(val).lower()

        #text = text.translate(trantab)

        tokens = nltk.word_tokenize(text)

        tokens = [word for word in tokens if word not in stopwords.words('english')]

        for words in tokens:

            words_corpus = words_corpus + words + ' '

    return words_corpus
pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))

neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))
def plot_Cloud(wordCloud):

    plt.figure( figsize=(20,10), facecolor='w')

    plt.imshow(wordCloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.show()
plot_Cloud(pos_wordcloud)
plot_Cloud(neg_wordcloud)
temp_df.Usefulness.value_counts()
sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'], data=temp_df)

plt.xlabel('Usefulness')

plt.show()
sns.countplot(x='Sentiment', hue='Usefulness', order=["positive", "negative"],hue_order=['>75%', '25-75%', '<25%'], data=temp_df)

plt.xlabel('Sentiment')

plt.show()
temp_df["text_word_count"] = temp_df["Text"].apply(lambda text: len(text.split()))

sns.boxplot(x='Score',y='text_word_count', data=temp_df, showfliers=False)

plt.show()
sns.violinplot(x='Usefulness', y='text_word_count', order=[">75%", "<25%"], data=temp_df)

plt.ylim(-50, 400)

plt.show()
x = temp_df.UserId.value_counts()

x.to_dict()

temp_df["reviewer_freq"] = temp_df["UserId"].apply(lambda counts: "Frequent (>50 reviews)" if x[counts]>50 else "Not Frequent (1-50)")
ax = sns.countplot(x='Score', hue='reviewer_freq', data=temp_df)

ax.set_xlabel('Score (Rating)')

plt.show()
y = temp_df[temp_df.reviewer_freq=="Frequent (>50 reviews)"].Score.value_counts()

z = temp_df[temp_df.reviewer_freq=="Not Frequent (1-50)"].Score.value_counts()

tot_y = y.sum()

y = (y/tot_y)*100

tot_z = z.sum()

z = (z/tot_z)*100
ax1 = plt.subplot(131)

y.plot(kind="bar",ax=ax1)

plt.xlabel("Score")

plt.ylabel("Percentage")

plt.title("Frequent (>50 reviews) Distribution")



ax2 = plt.subplot(133)

z.plot(kind="bar",ax=ax2)

plt.xlabel("Score")

plt.ylabel("Percentage")

plt.title("Not Frequent (1-50) Distribution")

plt.show()
sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'], hue='reviewer_freq', data=temp_df)

plt.xlabel('Helpfulness')

plt.show()
sns.violinplot(x='reviewer_freq', y='text_word_count', data=temp_df, palette='RdBu')

plt.xlabel('Frequency of Reviewer')

plt.ylim(-50, 400)

plt.show()