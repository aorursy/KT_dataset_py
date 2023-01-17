# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
obama_files = []

trump_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if "Obama" in filename and "sotu" in str(dirname):

            obama_files.append(os.path.join(dirname, filename))

        if "Trump" in filename and "sotu" in str(dirname):

            trump_files.append(os.path.join(dirname, filename))
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
def sentence_score(sentence):

    ss = sid.polarity_scores(sentence)

    return ss
rows = []

for file in obama_files:

    year = int(file[file.index("_")+1:file.index("_")+5])

    president = "Obama"

    with open(file, 'r') as f:

        lines = ' '.join(f.readlines())

        sent_text = nltk.sent_tokenize(lines)

        for sent in sent_text:

            if sent != "\n":

                ss = sentence_score(sent)

                rows.append([president, year, sent, ss['compound'], ss['pos'], ss['neu']])



for file in trump_files:

    year = int(file[file.index("_")+1:file.index("_")+5])

    president = "Trump"

    with open(file, 'r') as f:

        lines = ' '.join(f.readlines())

        sent_text = nltk.sent_tokenize(lines)

        for sent in sent_text:

            if sent != "\n":

                ss = sentence_score(sent)

                rows.append([president, year, sent, ss['compound'], ss['pos'], ss['neu']])

                
df = pd.DataFrame(columns=['president', 'year', 'sentence', 'compound', 'pos', 'neu'], data=rows)
df
obama_df = df[df.president=="Obama"]

trump_df = df[df.president=="Trump"]
sns.violinplot(data=df, x='year', y='compound')
sns.violinplot(data=df, x='year', y='pos')
sns.violinplot(data=df, x='year', y='neu')
df.groupby('year').mean().plot()
df.groupby(['president', 'year']).mean()
from nltk.corpus import stopwords

from collections import Counter
obama_years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

trump_years = [2017, 2018]
for year in obama_years:

    year_df = obama_df[obama_df['year'] == year]

    s = ' '.join(year_df['sentence'].tolist())

    words = s.split(' ')

    words = list(map(str.strip, words))

    words = list(map(str.lower, words))

    

    filtered_words = [word for word in words if word not in stopwords.words('english')]

    

    words_count = Counter(filtered_words).most_common()

    words_count = words_count[:30]

    

    words_df = pd.DataFrame(words_count)

    

    words_df.rename(columns={0:"Words", 1:"Count"}, inplace=True)

    

    fig, ax = plt.subplots()

    fig.set_size_inches(11.7, 8.27)

    fig.suptitle('Top 30 Words Used in Speech - Obama '+str(year), fontsize=20)

    sns.barplot(x='Words', y='Count', data=words_df, ax=ax)

    plt.xticks(rotation=80)

    plt.show()
for year in trump_years:

    year_df = trump_df[trump_df['year'] == year]

    s = ' '.join(year_df['sentence'].tolist())

    words = s.split(' ')

    words = list(map(str.strip, words))

    words = list(map(str.lower, words))

    

    filtered_words = [word for word in words if word not in stopwords.words('english')]

    

    words_count = Counter(filtered_words).most_common()

    words_count = words_count[:30]

    

    words_df = pd.DataFrame(words_count)

    

    words_df.rename(columns={0:"Words", 1:"Count"}, inplace=True)

    

    fig, ax = plt.subplots()

    fig.set_size_inches(11.7, 8.27)

    fig.suptitle('Top 30 Words Used in Speech - Trump '+str(year), fontsize=20)

    sns.barplot(x='Words', y='Count', data=words_df, ax=ax)

    plt.xticks(rotation=80)

    plt.show()