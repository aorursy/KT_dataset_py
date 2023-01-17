import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
df_you = pd.read_csv("../input/USvideos.csv")
# Understanding the bandwidth of the data
print(df_you.shape)
print(df_you.nunique())
#Looking for nulls and data types
df_you.info()
df_you.head(n=5)
df_you['likes_log'] = np.log(df_you['likes']+1)
df_you['views_log'] = np.log(df_you['views'] +1)
df_you['dislikes_log'] = np.log(df_you['dislikes'] +1)
df_you['comment_count_log'] = np.log(df_you['comment_count']+1)

plt.figure(figsize = (12,6))
plt.subplot(221)
g1 = sns.distplot(df_you['likes_log'], color = 'green')
g1.set_title("LIKES LOG DISTRIBUTION", fontsize = 16)

plt.subplot(222)
g2 = sns.distplot(df_you['views_log'])
g2.set_title("VIEWS LOG DISTRIBUTION", fontsize = 16)

plt.subplot(223)
g3 = sns.distplot(df_you['dislikes_log'], color = 'r')
g3.set_title("DISLIKES LOG DISTRIBUTION", fontsize=16)

plt.subplot(224)
g4 = sns.distplot(df_you['comment_count_log'])
g4.set_title("COMMENT COUNT LOG DISTRIBUTION", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4, top = 0.9)

plt.show()

# Let us find out the unique category ids present in our dataset
np.unique(df_you["category_id"])

df_you['category_name'] = np.nan

df_you.loc[(df_you["category_id"]== 1),"category_name"] = 'Film and Animation'
df_you.loc[(df_you["category_id"] == 2), "category_name"] = 'Cars and Vehicles'
df_you.loc[(df_you["category_id"] == 10), "category_name"] = 'Music'
df_you.loc[(df_you["category_id"] == 15), "category_name"] = 'Pet and Animals'
df_you.loc[(df_you["category_id"] == 17), "category_name"] = 'Sports'
df_you.loc[(df_you["category_id"] == 19), "category_name"] = 'Travel and Events'
df_you.loc[(df_you["category_id"] == 20), "category_name"] = 'Gaming'
df_you.loc[(df_you["category_id"] == 22), "category_name"] = 'People and Blogs'
df_you.loc[(df_you["category_id"] == 23), "category_name"] = 'Comedy'
df_you.loc[(df_you["category_id"] == 24), "category_name"] = 'Entertainment'
df_you.loc[(df_you["category_id"] == 25), "category_name"] = 'News and Politics'
df_you.loc[(df_you["category_id"] == 26), "category_name"] = 'How to and Style'
df_you.loc[(df_you["category_id"] == 27), "category_name"] = 'Education'
df_you.loc[(df_you["category_id"] == 28), "category_name"] = 'Science and Technology'
df_you.loc[(df_you["category_id"] == 29), "category_name"] = 'Non-profits and Activism'
df_you.loc[(df_you["category_id"] == 43), "category_name"] = 'Shows'

#Plotting the categories to see which ones are the popular ones 
plt.figure(figsize = (14,10))
g = sns.countplot('category_name', data = df_you, palette="Set1", order = df_you['category_name'].value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=45, ha="right")
g.set_title("Count of the Video Categories", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)
plt.subplots_adjust(wspace = 0.9, hspace = 0.9, top = 0.9)
plt.show()

#As is evident from the graph, we see that the top views categories are 'Entertainment', 'Music', 'How to and Style', 'Comedy' and 'People and Blogs'
#Showing the top 5 categories
print("Category Name count")
print(df_you['category_name'].value_counts()[:5])
plt.figure(figsize = (14,10))
g = sns.boxplot(x = 'category_name', y = 'views_log', data = df_you, palette="winter_r")
g.set_xticklabels(g.get_xticklabels(),rotation=45, ha="right")
g.set_title("Views across different categories", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Views(log)", fontsize=12)
plt.subplots_adjust(wspace = 0.9, hspace = 0.9, top = 0.9)
plt.show()
plt.figure(figsize = (14,10))
g = sns.boxplot(x = 'category_name', y = 'likes_log', data = df_you, palette="spring_r")
g.set_xticklabels(g.get_xticklabels(),rotation=45, ha="right")
g.set_title("Likes across different categories", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Likes(log)", fontsize=12)
plt.subplots_adjust(wspace = 0.9, hspace = 0.9, top = 0.9)
plt.show()
plt.figure(figsize = (14,10))
g = sns.boxplot(x = 'category_name', y = 'dislikes_log', data = df_you, palette="summer_r")
g.set_xticklabels(g.get_xticklabels(),rotation=45, ha="right")
g.set_title("Dislikes across different categories", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Dislikes(log)", fontsize=12)
plt.subplots_adjust(wspace = 0.9, hspace = 0.9, top = 0.9)
plt.show()
plt.figure(figsize = (14,10))
g = sns.boxplot(x = 'category_name', y = 'comment_count_log', data = df_you, palette="plasma")
g.set_xticklabels(g.get_xticklabels(),rotation=45, ha="right")
g.set_title("Comments count across different categories", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Comment_count(log)", fontsize=12)
plt.subplots_adjust(wspace = 0.9, hspace = 0.9, top = 0.9)
plt.show()
#Calculating the engagement measures - like rate, dislike rate and comment rate 
df_you['like_rate'] = df_you['likes']/df_you['views']
df_you['dislike_rate'] = df_you['dislikes']/df_you['views']
df_you['comment_rate'] = df_you['comment_count']/df_you['views']
#Building a correlation matrix using a heatmap for engagement measures
plt.figure(figsize = (10,8))
sns.heatmap(df_you[['like_rate', 'dislike_rate', 'comment_rate']].corr(), annot=True)
plt.show()
import string
import re 
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer   
#Word count 
df_you['count_word']=df_you['title'].apply(lambda x: len(str(x).split()))
df_you['count_word_tags']=df_you['tags'].apply(lambda x: len(str(x).split()))

#Unique word count
df_you['count_unique_word'] = df_you['title'].apply(lambda x: len(set(str(x).split())))
df_you['count_unique_word_tags'] = df_you['tags'].apply(lambda x: len(set(str(x).split())))

#Punctutation count
df_you['count_punctuation'] = df_you['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_you['count_punctuation_tags'] = df_you['tags'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#Average length of the words
df_you['mean_word_len'] = df_you['title'].apply(lambda x : np.mean([len(x) for x in str(x).split()]))
df_you['mean_word_len_tags'] = df_you['tags'].apply(lambda x: np.mean([len(x) for x in str(x).split()]))
#Deriving unique word count percent in each comment
df_you['word_unique_percent'] = df_you['count_unique_word']*100/df_you['count_word']
df_you['word_unique_percent_tags'] = df_you['count_unique_word_tags']*100/df_you['count_word_tags']

#Deriving the punctuation rate in each comment
df_you['punct_percent'] = df_you['count_punctuation']*100/df_you['count_word']
df_you['punct_percent_tags'] = df_you['count_punctuation_tags']*100/df_you['count_word_tags']

plt.figure(figsize = (12,18))

plt.subplot(421)
g1 = sns.distplot(df_you['count_word'],
                 hist = False, label = 'Text')
g1 = sns.distplot(df_you['count_word_tags'],
                 hist = False, label = 'Tags')
g1.set_title('Word count distribution', fontsize = 14)
g1.set(xlabel='Word Count')

plt.subplot(422)
g2 = sns.distplot(df_you['count_unique_word'],
                 hist = False, label = 'Text')
g2 = sns.distplot(df_you['count_unique_word_tags'],
                 hist = False, label = 'Tags')
g2.set_title('Unique word count distribution', fontsize = 14)
g2.set(xlabel='Unique Word Count')

plt.subplot(423)
g3 = sns.distplot(df_you['count_punctuation'],
                 hist = False, label = 'Text')
g3 = sns.distplot(df_you['count_punctuation_tags'],
                 hist = False, label = 'Tags')
g3.set_title('Punctuation count distribution', fontsize =14)
g3.set(xlabel='Punctuation Count')

plt.subplot(424)
g4 = sns.distplot(df_you['mean_word_len'],
                 hist = False, label = 'Text')
g4 = sns.distplot(df_you['mean_word_len_tags'],
                 hist = False, label = 'Tags')
g4.set_title('Average word length distribution', fontsize = 14)
g4.set(xlabel = 'Average Word Length')

plt.subplots_adjust(wspace = 0.2, hspace = 0.4, top = 0.9)
plt.legend()
plt.show()
# Creating a heatmap to explore correlation between word count, unique word count, punctuation count and average word length

plt.figure(figsize = (10,8))
sns.heatmap(df_you[['count_word','count_word_tags','count_unique_word','count_unique_word_tags','count_punctuation','count_punctuation_tags','mean_word_len','mean_word_len_tags']].corr(), annot = True)
plt.show()

plt.figure(figsize = (20,20))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                      background_color = 'black',
                      stopwords=stopwords,
                      max_words = 1000,
                      max_font_size = 120,
                      random_state = 42
                    ).generate(str(df_you['title']))

#Plotting the word cloud
plt.imshow(wordcloud)
plt.title("WORD CLOUD for Titles", fontsize = 20)
plt.axis('off')
plt.show()
plt.figure(figsize = (20,20))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                      background_color = 'black',
                      stopwords = stopwords,
                      max_words = 1000,
                      max_font_size = 120,
                      random_state = 42
                    ).generate(str(df_you['description']))

plt.imshow(wordcloud)
plt.title('WORD CLOUD for Title Description', fontsize = 20)
plt.axis('off')
plt.show()
plt.figure(figsize = (20,20))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                      background_color = 'black',
                      stopwords = stopwords,
                      max_words = 1000,
                      max_font_size = 120,
                      random_state = 42
                    ).generate(str(df_you['tags']))

plt.imshow(wordcloud)
plt.title('WORD CLOUD for Tags', fontsize = 20)
plt.axis('off')
plt.show()