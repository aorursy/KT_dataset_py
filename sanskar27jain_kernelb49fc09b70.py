# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize.regexp import WordPunctTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/raw_text.csv")

df.head()
tags = list(df.label.unique())

print(tags)
index_to_tags_dict = {i:tag for i,tag in enumerate(tags)}

tags_to_index_dict = {tag:i for i,tag in enumerate(tags)}

print(index_to_tags_dict)

print(tags_to_index_dict)
bag_of_words = {}

sentence_tokens = []

for text in df.text:

    tokens = WordPunctTokenizer().tokenize(text)

    token_list = []

    for token in tokens:

        token = token.lower()

        if token not in '''!()-[]{};:'"\,<>./?@#$%^&*_~''':

            try:

                int(token)

                continue

            except:

                token_list.append(token)

                bag_of_words[token] = bag_of_words.get(token,0)+1

    sentence_tokens.append(token_list)



print(len(bag_of_words))
# Taking top n words

n = 10000

top_n_words = sorted(bag_of_words.items(), key = lambda item: item[1], reverse = True)[:n]

print(top_n_words)

top_n_words_to_index = {item[0]:i for i,item in enumerate(top_n_words)}

print(top_n_words_to_index)
# Processed Subtitles

subtitles = []

for token_list in sentence_tokens:

    sub = []

    for token in token_list:

        if top_n_words_to_index.get(token,-1) != -1:

            sub.append(token)

    sub = " ".join(sub)

    subtitles.append(sub)

print(subtitles[0])
# Word Binary, Word Count and Tfidf Features for each sentence

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(subtitles)

word_count_features = np.array(X.toarray())

word_binary_features = np.array(word_count_features>0, dtype = int)



vectorizer2 = TfidfVectorizer()

X2 =  vectorizer2.fit_transform(subtitles)

tfidf_features = np.array(X2.toarray())



print(word_count_features.shape)

print(word_binary_features.shape)

print(word_binary_features[0])

print(word_count_features[0])

print(tfidf_features[0])
labels = [tags_to_index_dict[label] for label in df.label]

print(labels)
train = 700
# Multiclass Logistic Classifier for Word Binary Features

clf_logistic_wb = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=10000)

clf_logistic_wb = clf_logistic_wb.fit(word_binary_features[:train], labels[:train])

pred_logistic_wb = clf_logistic_wb.predict(word_binary_features[train:])

accuracy_logistic_wb = np.mean(pred_logistic_wb==labels[train:])*100

print("Accuracy =",accuracy_logistic_wb)
# Multiclass SVM Classifier for Word Binary Features

clf_svm_wb = svm.SVC(gamma='scale')

clf_svm_wb = clf_svm_wb.fit(word_binary_features[:train], labels[:train])

pred_svm_wb = clf_svm_wb.predict(word_binary_features[train:])

accuracy_svm_wb = np.mean(pred_svm_wb==labels[train:])*100

print("Accuracy =",accuracy_svm_wb)
# Mutliclass Naive Bayes Classifier for Word Binary Features

clf_nb_wb = GaussianNB()

clf_nb_wb = clf_nb_wb.fit(word_binary_features[:train], labels[:train])

pred_nb_wb = clf_nb_wb.predict(word_binary_features[train:])

accuracy_nb_wb = np.mean(pred_nb_wb==labels[train:])*100

print("Accuracy =",accuracy_nb_wb)
# Multiclass Logistic Classifier for Word Count Features

clf_logistic_wc = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=2000)

clf_logistic_wc = clf_logistic_wc.fit(word_count_features[:train], labels[:train])

pred_logistic_wc = clf_logistic_wc.predict(word_count_features[train:])

accuracy_logistic_wc = np.mean(pred_logistic_wc==labels[train:])*100

print("Accuracy =",accuracy_logistic_wc)
# Multiclass SVM Classifier for Word Count Features

clf_svm_wc = svm.SVC(gamma='scale')

clf_svm_wc = clf_svm_wc.fit(word_count_features[:train], labels[:train])

pred_svm_wc = clf_svm_wc.predict(word_count_features[train:])

accuracy_svm_wc= np.mean(pred_svm_wc==labels[train:])*100

print("Accuracy =",accuracy_svm_wc)
# Mutliclass Naive Bayes Classifier for Word Count Features

clf_nb_wc = GaussianNB()

clf_nb_wc = clf_nb_wc.fit(word_count_features[:train], labels[:train])

pred_nb_wc = clf_nb_wc.predict(word_count_features[train:])

accuracy_nb_wc = np.mean(pred_nb_wc==labels[train:])*100

print("Accuracy =",accuracy_nb_wc)
# Multiclass Logistic Classifier for Tfidf Features

clf_logistic_tfidf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=2000)

clf_logistic_tfidf = clf_logistic_tfidf.fit(tfidf_features[:train], labels[:train])

pred_logistic_tfidf = clf_logistic_tfidf.predict(tfidf_features[train:])

accuracy_logistic_tfidf = np.mean(pred_logistic_tfidf==labels[train:])*100

print("Accuracy =",accuracy_logistic_tfidf)
# Multiclass SVM Classifier for Word Tfidf Features

clf_svm_tfidf = svm.SVC(gamma='scale')

clf_svm_tfidf = clf_svm_tfidf.fit(tfidf_features[:train], labels[:train])

pred_svm_tfidf = clf_svm_tfidf.predict(tfidf_features[train:])

accuracy_svm_tfidf= np.mean(pred_svm_tfidf==labels[train:])*100

print("Accuracy =",accuracy_svm_tfidf)
# Mutliclass Naive Bayes Classifier for Tfidf Features

clf_nb_tfidf = GaussianNB()

clf_nb_tfidf = clf_nb_tfidf.fit(tfidf_features[:train], labels[:train])

pred_nb_tfidf = clf_nb_tfidf.predict(tfidf_features[train:])

accuracy_nb_tfidf = np.mean(pred_nb_tfidf==labels[train:])*100

print("Accuracy =",accuracy_nb_tfidf)