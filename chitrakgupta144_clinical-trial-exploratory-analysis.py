import pandas as pd

df = pd.read_csv("../input/clinical_trial.csv")

print(df.shape)

print(sum(df['abstract'].isnull()), " records have no abstract")

print(sum(df['title'].isnull()), " records have no title")

print(df['trial'].value_counts())
# First, replace the "NA" in abstract by blank

df['abstract'] = df['abstract'].fillna(value="")

df['title_length'] = df['title'].str.split(' ').str.len()

df['abstract_length'] = df['abstract'].str.split(' ').str.len()

import seaborn as sns

sns.factorplot(x="title_length", y="abstract_length", hue='trial', data=df, kind='swarm')

sns.plt.show()
import nltk

def get_most_freq_word(series, top_N = 10):

    formatted = series.str.lower().str.replace("'","").str.replace('[^\w\s]',' ').str.replace('[^\D]',' ').str.cat(sep=' ')

    words = nltk.tokenize.word_tokenize(formatted)

    word_dist = nltk.FreqDist(words)

    stopwords = nltk.corpus.stopwords.words('english')

    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords)

    if (top_N > 0):

        return dict(words_except_stop_dist.most_common(top_N))

    else:

        return dict(words_except_stop_dist)
# Most frequent words in clinical trial paper titles

print(get_most_freq_word(df[df['trial'] == 1]['title']))

# Most frequent words in non clinical trial paper titles

print(get_most_freq_word(df[df['trial'] == 0]['title']))
# Most frequent words in clinical trial paper titles

print(get_most_freq_word(df[df['trial'] == 1]['title'], top_N = 20))

# Most frequent words in non clinical trial paper titles

print(get_most_freq_word(df[df['trial'] == 0]['title'], top_N = 20))
top20_trial = get_most_freq_word(df[df['trial'] == 1]['title'], top_N = 20)

labels = sorted(top20_trial, key = top20_trial.get, reverse=True)

values = sorted(top20_trial.values(), reverse=True)

g = sns.barplot(x=labels, y=values)

g.set_xticklabels(labels, rotation=30)

sns.plt.gcf().subplots_adjust(bottom=0.2)

sns.plt.show()
def getUnique(text):

    unique_words = []

    wordsList = text.split()

    for word in wordsList:

        if word not in unique_words:

            unique_words.append(word)

            

    return unique_words



allWords = get_most_freq_word(df['title'], top_N = -1)

bow = allWords.keys()
from collections import defaultdict

trialList = defaultdict(int)

notTrialList = defaultdict(int)

for word in bow:

    trialList[word] += 0.5

    notTrialList[word] += 0.5

    

trial = df[df['trial'] == 1]

for i in range(trial.shape[0]):

    wordCounts = get_most_freq_word(pd.Series(trial["title"].values[i]), top_N = -1)

    uniqueWords = wordCounts.keys()

    for word in uniqueWords:

        trialList[word] += 1

        

notTrial = df[df['trial'] == 0]

for i in range(notTrial.shape[0]):

    wordCounts = get_most_freq_word(pd.Series(notTrial["title"].values[i]), top_N = -1)

    uniqueWords = wordCounts.keys()

    for word in uniqueWords:

        notTrialList[word] += 1
# Create word table

wordTable = pd.DataFrame([trialList, notTrialList]).T

wordTable.columns = ["Trial", "Not Trial"]

wordTable["Trial"] = wordTable["Trial"]/sum(wordTable["Trial"] > 0.5)

wordTable["Not Trial"] = wordTable["Not Trial"]/sum(wordTable["Not Trial"] > 0.5)

import numpy as np

wordTable["presentLogOdds"] = np.log(wordTable["Trial"]) - np.log(wordTable["Not Trial"])

wordTable["absentLogOdds"] = np.log(1-wordTable["Trial"]) - np.log(1-wordTable["Not Trial"])