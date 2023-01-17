# initial setup



import numpy as np

import pandas as pd

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()
# check the input data



for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + ' MB')
# check the fields in the dataset



df_train = pd.read_csv('../input/train.csv')

df_train.head()
print('Total number of question pairs for training: {}'.format(len(df_train)))

print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(

    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
# basic statistics



print('Total number of question pairs for training: {}'.format(len(df_train)))

print('Postive Class (Duplicate pairs): {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(

    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))



plt.figure(figsize=(12, 5))

plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print()
# check the fields in the test set



df_test = pd.read_csv('../input/test.csv')

df_test.head()
print('Total number of question pairs for testing: {}'.format(len(df_test)))
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)



dist_train = train_qs.apply(len)

dist_test = test_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')

plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')

plt.title('Normalised histogram of character count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 

                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
# naive method for splitting words (splitting on spaces instead of using a serious tokenizer):

dist_train = train_qs.apply(lambda x: len(x.split(' ')))

dist_test = test_qs.apply(lambda x: len(x.split(' ')))



plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')

plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')

plt.title('Normalised histogram of word count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 

                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
qmarks = np.mean(train_qs.apply(lambda x: '?' in x))

math = np.mean(train_qs.apply(lambda x: '[math]' in x))

fullstop = np.mean(train_qs.apply(lambda x: '.' in x))

capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))

capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))

numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))



print('Questions with question marks: {:.2f}%'.format(qmarks * 100))

print('Questions with [math] tags: {:.2f}%'.format(math * 100))

print('Questions with full stops: {:.2f}%'.format(fullstop * 100))

print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))

print('Questions with capital letters: {:.2f}%'.format(capitals * 100))

print('Questions with numbers: {:.2f}%'.format(numbers * 100))
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R



plt.figure(figsize=(15, 5))

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over fraction of words shared by questions pairs', fontsize=15)

plt.xlabel('Fraction of words shared', fontsize=15)