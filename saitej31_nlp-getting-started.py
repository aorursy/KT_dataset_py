import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import re

import string



import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print(len(train_data))

print(len(test_data))
y = train_data.target

train_data =train_data.drop(['id','keyword','location','target'],axis =1)
def process_text(text):

    stemmer = PorterStemmer()

    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE

    text = re.sub(r'\$\w*', '', text)

    # remove old style retweet text "RT"

    text = re.sub(r'^RT[\s]+', '', text)

    # remove hyperlinks

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # remove hashtags

    # only removing the hash # sign from the word

    text = re.sub(r'#', '', text)

    # tokenize tweets

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,

                               reduce_len=True)

    text_tokens = tokenizer.tokenize(text)



    texts_clean = []

    for word in text_tokens:

        if (word not in stopwords_english and  # remove stopwords

                word not in string.punctuation):  # remove punctuation

            # tweets_clean.append(word)

            stem_word = stemmer.stem(word)  # stemming word

            texts_clean.append(stem_word)





    return texts_clean

def build_freqs(texts, ys):



    yslist = np.squeeze(ys).tolist()



    freqs = {}

    for y, text in zip(yslist, texts):

        for word in process_text(text):

            pair = (word, y)

            if pair in freqs:

                freqs[pair] += 1

            else:

                freqs[pair] = 1



    return freqs
freqs = build_freqs(train_data['text'],y)
freqs
def extract_features(text, freqs):

    # process_tweet tokenizes, stems, and removes stopwords

    word_l = process_text(text)

    

    # 3 elements in the form of a 1 x 3 vector

    x = np.zeros((1, 3)) 

    

    #bias term is set to 1

    x[0,0] = 1 

    

    

    # loop through each word in the list of words

    for word in word_l:

        

        # increment the word count for the positive label 1

        x[0,1] += freqs.get((word,1.0), 0)

        

        # increment the word count for the negative label 0

        x[0,2] += freqs.get((word,0.0), 0)

        

    assert(x.shape == (1, 3))

    return x
X = np.zeros((len(train_data), 3))

for i in range(len(train_data)):

    X[i, :]= extract_features(train_data.text[i], freqs)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score

y.value_counts()
len(X)
model = LogisticRegression(penalty= 'l2' ,random_state= 42 ,max_iter=20,solver='liblinear',class_weight= 'balanced')

model.fit(X[:5500],y[:5500])
y_pred = model.predict(X[5500:])
plot_confusion_matrix(model, X[:5500], y[:5500],labels=[0,1],normalize= 'true')
accuracy = accuracy_score(y[5500:],y_pred)

accuracy
main_model = LogisticRegression(penalty= 'l2' ,random_state= 42 ,max_iter=20,solver='liblinear',class_weight= 'balanced')

main_model.fit(X,y)
X_test = np.zeros((len(test_data), 3))

for i in range(len(test_data)):

    X_test[i, :]= extract_features(test_data.text[i], freqs)

y_test_pred = main_model.predict(X_test)
submission = pd.DataFrame({'id':sample['id'],'target': y_test_pred})

submission.to_csv('My_submission.csv',index = False)