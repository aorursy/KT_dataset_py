%matplotlib inline

import re

import nltk

import string

import random

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
# nltk.download(['twitter_samples', 'stopwords'])
from colorama import init, Fore, Style

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer

from nltk.corpus import twitter_samples, stopwords
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
positive_twts = twitter_samples.strings('positive_tweets.json')

negative_twts = twitter_samples.strings('negative_tweets.json')
print(Fore.GREEN + 'Positive Tweets: ' + str(len(positive_twts)))

print(Fore.RED + 'Negative Tweets: ' + str(len(negative_twts)))
fig, ax = plt.subplots(figsize=(5, 5))

ax.pie(

    [len(positive_twts), len(negative_twts)],

    labels=['Positive', 'Negative'],

    autopct='%2.2f%%',

    startangle=90

)

plt.show()
print(Fore.GREEN + positive_twts[random.randint(0, 5000)])

print(Fore.RED + negative_twts[random.randint(0, 5000)])
print(stopwords.words('english'))
print(string.punctuation)
def process_tweet(tweet):

    """

    Preproceeses a Tweet by removing hashes, RTs, @mentions,

    links, stopwords and punctuation, tokenizing and stemming 

    the words.

    

    Accepts:

        tweet {str} -- tweet string

    

    Returns:

        {list<str>}

    """

    

    proc_twt = re.sub(r'^RT[\s]+', '', tweet)

    proc_twt = re.sub(r'@[\w_-]+', '', proc_twt)

    proc_twt = re.sub(r'#', '', proc_twt)

    proc_twt = re.sub(r'https?:\/\/.*[\r\n]*', '', proc_twt)

    tokenizer = TweetTokenizer(

        preserve_case=False, 

        strip_handles=True,

        reduce_len=True

    )

    

    twt_clean = []

    twt_tokens = tokenizer.tokenize(proc_twt)

    stopwords_en = stopwords.words('english')

    for word in twt_tokens:

        if word not in stopwords_en and word not in string.punctuation:

            twt_clean.append(word)

            

    twt_stems = []

    stemmer = PorterStemmer()

    for word in twt_clean:

        twt_stems.append(stemmer.stem(word))

        

    return twt_stems
def build_freqs(tweets, labels):

    """

    Builds frequencies for a set of twwets and

    matching labels or sentiments

    

    Accepts:

        tweets {list<str>} -- tweets in the dataset

        labels {float} -- label (0. = -ve; 1. = +ve)

    

    Returns:

        {list<float>}

    """

        

    labels = np.squeeze(labels).tolist()



    freqs = {}

    for (tweet, label) in zip(tweets, labels):

        for word in process_tweet(tweet):

            if (word, label) in freqs:

                freqs[(word, label)] += 1

            else:

                freqs[(word, label)] = 1

                

    return freqs
labels = np.append(

    np.ones(len(positive_twts)),

    np.zeros(len(negative_twts))

)

labels
freqs = build_freqs(positive_twts + negative_twts, labels)
keys = [

    'happi', 'merri', 'nice', 'good', 

    'bad','sad', 'mad', 'best', 'pretti',

    '❤', ':)', ':(', '😒', '😬', '😄',

    '😍', '♛', 'song', 'idea', 'power', 

    'play', 'magnific', 'grind', 'bruise'

]



data = []



for word in keys:

    pos_freq = neg_freq = 0

    if (word, 1) in freqs:

        pos_freq = freqs[(word, 1)]

    if (word, 0) in freqs:

        neg_freq = freqs[(word, 0)]

    data.append([word, pos_freq, neg_freq])

    

data
fig, ax = plt.subplots(figsize = (8, 8))



x = np.log([x[1] + 1 for x in data])  

y = np.log([x[2] + 1 for x in data]) 



ax.scatter(x, y)  



plt.xlabel("Log Positive Count")

plt.ylabel("Log Negative Count")



for i in range(0, len(data)):

    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)



ax.plot([0, 9], [0, 9], color='red')



plt.show()
def extract_features(tweet, freqs):

    '''

    Input: 

        tweet: a list of words for one tweet

        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)

    Output: 

        x: a feature vector of dimension (1, 2)

    '''

    word_l = process_tweet(tweet)

    x = np.zeros((1, 2)) 

    for word in word_l:

        x[0, 0] += freqs.get((word, 1), 0)

        x[0, 1] += freqs.get((word, 0), 0)

    assert(x.shape == (1, 2))

    return x
extract_features('This is a sample tweet in which I am happy :)', freqs)
x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(

    positive_twts, labels[:5000],

    test_size=0.2, random_state=42

)
x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(

    negative_twts, labels[5000:],

    test_size=0.2, random_state=42

)
x_train = x_train_pos + x_train_neg

y_train = np.append(np.ones(len(x_train_pos)), np.zeros(len(x_train_neg)))
x_test = x_test_pos + x_test_neg

y_test = np.append(np.ones(len(x_test_pos)), np.zeros(len(x_test_neg)))
print(Fore.GREEN + 'Length of training set: ', str(len(x_train)))

print(Fore.RED + 'Length of testing set: ' + str(len(x_test)))
print(Fore.GREEN + 'Shape of training set labels: ', str(y_train.shape))

print(Fore.RED + 'Shape of testing set labels: ' + str(y_test.shape))
fig = plt.figure(figsize=(5, 5))

plt.scatter(

    np.array([extract_features(x, freqs)[0] for x in x_train])[:, 1],

    np.array([extract_features(x, freqs)[0] for x in x_train])[:, 0],

    c=[['green', 'red'][int(y)] for y in y_train],

    s=1

)

plt.show()
fig = plt.figure(figsize=(5, 5))

plt.scatter(

    np.array([extract_features(x, freqs)[0] for x in x_test])[:, 1],

    np.array([extract_features(x, freqs)[0] for x in x_test])[:, 0],

    c=[['green', 'red'][int(y)] for y in y_test],

    s=1

)

plt.show()
x_lr_train = np.array([extract_features(x, freqs)[0] for x in x_train])

y_lr_train = y_train



x_lr_test = np.array([extract_features(x, freqs)[0] for x in x_test])

y_lr_test = y_test
from sklearn.linear_model import SGDClassifier



logRegModel = SGDClassifier(loss='log')

logRegModel.fit(x_lr_train, y_lr_train)



y_lr_pred = logRegModel.predict(x_lr_test)
print('Confusion Matrix:')

sns.heatmap(confusion_matrix(y_test, y_lr_pred), cmap='YlGnBu')

plt.show()
print('Classification Report:')

print(classification_report(y_test, y_lr_pred, target_names=['-ve', '+ve']))
acc_logReg = accuracy_score(y_test, y_lr_pred) * 100

print(f'Accuracy: {acc_logReg:2.2f}%')
x_nb_train = np.array([extract_features(x, freqs)[0] for x in x_train])

y_nb_train = y_train



x_nb_test = np.array([extract_features(x, freqs)[0] for x in x_test])

y_nb_test = y_test
from sklearn.naive_bayes import CategoricalNB



nbModel = CategoricalNB()

nbModel.fit(x_nb_train, y_nb_train)



y_nb_pred = nbModel.predict(x_nb_test)
print('Confusion Matrix:')

sns.heatmap(confusion_matrix(y_test, y_nb_pred), cmap='YlGnBu')

plt.show()
print('Classification Report:')

print(classification_report(y_test, y_nb_pred, target_names=['-ve', '+ve']))
acc_nb = accuracy_score(y_test, y_nb_pred) * 100

print(f'Accuracy: {acc_nb:2.2f}%')
x_knn_train = np.array([extract_features(x, freqs)[0] for x in x_train])

y_knn_train = y_train



x_knn_test = np.array([extract_features(x, freqs)[0] for x in x_test])

y_knn_test = y_test
from sklearn.neighbors import KNeighborsClassifier



kMax = 20

kVals = list(range(1, kMax + 1))

mean_acc = np.zeros(len(kVals))

std_acc = np.zeros(len(kVals))



for i in kVals:

    knnModel = KNeighborsClassifier(n_neighbors=i).fit(x_knn_train, y_knn_train)

    yHat = knnModel.predict(x_knn_test)

    mean_acc[i - 1] = np.mean(yHat == y_knn_test);

    

bestK = pd.DataFrame({'k':kVals, 'mean_acc':mean_acc}).set_index('k')['mean_acc'].idxmax()

print(Fore.YELLOW + 'Best k =', bestK)
knnModel = KNeighborsClassifier(n_neighbors=bestK).fit(x_knn_train, y_knn_train)



y_knn_pred = knnModel.predict(x_knn_test)
print('Confusion Matrix:')

sns.heatmap(confusion_matrix(y_test, y_knn_pred), cmap='YlGnBu')

plt.show()
print('Classification Report:')

print(classification_report(y_test, y_knn_pred, target_names=['-ve', '+ve']))
acc_knn = accuracy_score(y_test, y_knn_pred) * 100

print(f'Accuracy: {acc_knn:2.2f}%')
ax = sns.barplot(['LogReg', 'NaiveBayes', 'KNN'], [acc_logReg, acc_nb, acc_knn])

for p in ax.patches:

    ax.annotate(

        f'{p.get_height():2.2f}%', 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, -20), textcoords = 'offset points'

    )

plt.xlabel('Models')

plt.ylabel('Accuracy')

plt.show()