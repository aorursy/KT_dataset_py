import os
import time
import numpy as np
import pandas as pd 
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/train.csv")
train_df.shape
train_df.head(10)
target_count = train_df.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');
import nltk
from nltk.corpus import stopwords

def tokenizer(file_text):
    tokens = nltk.word_tokenize(file_text)

    stop_words = stopwords.words('english')
    tokens = [i for i in tokens if ( i not in stop_words )]
    
    return ' '.join(tokens)

train_df.question_text = train_df.question_text.apply(lambda x: tokenizer(x))

train_df.head(10)
text = ' '.join(train_df['question_text'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, background_color='black',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in question text')
plt.axis("off")
plt.show()
train_df[train_df['target']==0].question_text.head(10)
text = ' '.join(train_df[train_df['target']==0].question_text.str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, background_color='black',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in nontoxic question text')
plt.axis("off")
plt.show()
text = ' '.join(train_df[train_df['target']==1].question_text.str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, background_color='black',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in toxic question text')
plt.axis("off")
plt.show()
pd.set_option('display.max_colwidth', -1)
train_df[(train_df['target']==1) & (train_df['question_text'].str.contains("Trump"))].head(10)
train_df[(train_df['target']==0) & (train_df['question_text'].str.contains("Trump"))].head(10)