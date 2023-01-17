# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# nltk

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import dataset

q_quora = pd.read_csv("../input/first-quora-dataset/q_quora.csv")

q_quora = q_quora[['id','qid1','qid2','question1','question2','is_duplicate']]

q_quora.head()
# number of rows and columns

q_quora.shape
# label distribution

q_quora.is_duplicate.value_counts()
# data types

q_quora.dtypes
q_quora_clean = q_quora[(q_quora['is_duplicate'] == "0") | (q_quora['is_duplicate'] == "1")]
label_distri = q_quora_clean.is_duplicate.value_counts()



plt.figure(figsize=(8,4))

sns.barplot(label_distri.index, label_distri.values, alpha=0.8)

plt.title("The distribution of Label")

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Duplicate', fontsize=12)

plt.show()



label_distri / label_distri.sum()
df_all_questions = pd.DataFrame(pd.concat([q_quora_clean['question1'], q_quora_clean['question2']]))

df_all_questions.columns = ['questions']

df_all_questions = df_all_questions.reset_index(drop=True)

# word count

df_all_questions['word_counts'] = df_all_questions['questions'].apply(lambda x: len(str(x).split()))
word_count_distri = df_all_questions['word_counts'].value_counts()



plt.figure(figsize=(32,18))

sns.barplot(word_count_distri.index, word_count_distri.values, alpha=0.8)

plt.title('Distribution of word counts')

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('word counts in the question', fontsize=12)

plt.show()
df_all_questions['character_counts'] = df_all_questions['questions'].apply(lambda x: len(str(x)))

charac_counts_dist = df_all_questions['character_counts'].value_counts()



plt.figure(figsize=(40,10))

sns.barplot(charac_counts_dist.index, charac_counts_dist.values, alpha=0.8)

plt.title('Distribution of character counts')

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('character counts in the question', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
# common english stop words

eng_stopwords = set(stopwords.words('english'))



def get_unigrams(question):

    return [word for word in word_tokenize(question.lower()) if word not in eng_stopwords]



def get_common_unigrams(row):

    return len(set(row['unigram_ques1']).intersection(set(row['unigram_ques2'])))



def get_common_unigram_ratio(row):

    return row["unigrams_common_count"] / max(len(set('unigrams_ques1').union(set('unigrams_ques2'))),1)
q_quora_clean['unigram_ques1'] = q_quora_clean['question1'].apply(lambda x: get_unigrams(str(x)))

q_quora_clean['unigram_ques2'] = q_quora_clean['question2'].apply(lambda x: get_unigrams(str(x)))

q_quora_clean['unigrams_common_count'] = q_quora_clean.apply(lambda row: get_common_unigrams(row),axis=1)

q_quora_clean['unigrams_common_ratio'] = q_quora_clean.apply(lambda row: get_common_unigram_ratio(row), axis=1)
unigrams_count = q_quora_clean['unigrams_common_count'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(unigrams_count.index, unigrams_count.values, alpha=0.8)

plt.title('Distribution of Unigrams Count')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Common unigrams count', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.violinplot(x="is_duplicate", y="unigrams_common_count", data=q_quora_clean, palette="muted")

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigrams count', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.violinplot(x="is_duplicate", y="unigrams_common_ratio", data=q_quora_clean)

plt.ylim(0,1)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigrams ratio', fontsize=12)

plt.show()
def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))



q_quora_clean['q1len'] = q_quora_clean['question1'].str.len()

q_quora_clean['q2len'] = q_quora_clean['question2'].str.len()



q_quora_clean['q1_n_words'] = q_quora_clean['question1'].apply(lambda row: len(str(row).split(" ")))

q_quora_clean['q2_n_words'] = q_quora_clean['question2'].apply(lambda row: len(str(row).split(" ")))
def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))



q_quora_clean['word_share'] = q_quora_clean.apply(normalized_word_share, axis=1)
plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'word_share', data = q_quora_clean[0:50000])

plt.subplot(1,2,2)

sns.distplot(q_quora_clean[q_quora_clean['is_duplicate'] == '1']['word_share'][0:10000], color = 'green')

sns.distplot(q_quora_clean[q_quora_clean['is_duplicate'] == '0']['word_share'][0:10000], color = 'orange')
