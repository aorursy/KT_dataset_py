# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import seaborn as sns
import re
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
train_data = pd.read_csv('../input/train.csv')
print("Size of training data:",train_data.size)
print("Columns in the traing data",train_data.columns)
print("question_text ->",train_data.iloc[1]['question_text'],"Target:->",train_data.iloc[1]['target'])
target_count = train_data[['qid','target']].groupby(['target']).agg('count').reset_index()
target_count.columns = ["target","Count"]
labels = ['Sincere Questions','Insincere Questions']  
fig1, ax1 = plt.subplots()
colors = ['#66b3ff','#ff9999']
ax1.pie(target_count["Count"],labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()
print("total sincere questions",target_count.iloc[0]["Count"])
print("total insincere questions",target_count.iloc[1]["Count"])
insincere_question_df = train_data[['question_text']][train_data['target'] == 1]
sincere_question_df = train_data[['question_text']][train_data['target'] == 0]
insincere_question_df.size
def text_clean(review_col):
    stops = set(stopwords.words("english"))
    stops.update(['would','many','u','much','more'])
    text_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        word_token = word_tokenize(str(review).lower())
        #review = [word for word in word_token if word not in stops]
        #review=' '.join(review)
        review=[lemma.lemmatize(w) for w in word_token if w not in stops]
        review=' '.join(review)
        text_corpus.append(review) 
    return text_corpus
insincere_question_df['question_word']=text_clean(insincere_question_df['question_text'].values)
insincere_question_clean_word = insincere_question_df.question_word.str.split(expand=True).stack().value_counts().to_frame()
insincere_question_clean_word.reset_index(inplace=True)
insincere_question_clean_word.columns = ['word','count']
sns.set(style="white")
# Plot word and it's count for top 50 words
sns.relplot(x="count", y="word", size="count",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=8, data=insincere_question_clean_word[0:20])
sincere_question_df['question_word']=text_clean(sincere_question_df['question_text'].values)
sincere_question_clean_word = sincere_question_df.question_word.str.split(expand=True).stack().value_counts().to_frame()
sincere_question_clean_word.reset_index(inplace=True)
sincere_question_clean_word.columns = ['word','count']
sns.set(style="white")
# Plot word and it's count for top 50 words
sns.relplot(x="count", y="word", size="count",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=8, data=sincere_question_clean_word[0:20])
