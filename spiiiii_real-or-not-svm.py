from collections import defaultdict

import os



import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords

from nltk import pos_tag



from collections import defaultdict

import os
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_df.text = train_df.text.str.lower()

train_df.text =  train_df.text.apply(word_tokenize)



test_df.text = test_df.text.str.lower()

test_df.text =  test_df.text.apply(word_tokenize)
tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

for index,entry in enumerate(test_df['text']):

    Final_words = []

    word_Lemmatized = WordNetLemmatizer()

    for word, tag in pos_tag(entry):

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words.append(word_Final)

    test_df.loc[index,'text_final'] = str(Final_words)



for index,entry in enumerate(train_df['text']):

    Final_words = []

    word_Lemmatized = WordNetLemmatizer()

    for word, tag in pos_tag(entry):

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words.append(word_Final)

    train_df.loc[index,'text_final'] = str(Final_words)

train_df.text_final
Tfidf_vect = TfidfVectorizer(max_features=12000)

Tfidf_vect.fit(train_df['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(train_df['text_final'])

Test_X_Tfidf = Tfidf_vect.transform(test_df['text_final'])



Train_X_Tfidf
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_df.target)
predictions_SVM = SVM.predict(Test_X_Tfidf)
type(predictions_SVM)
subm = pd.concat([test_df.id, pd.Series(predictions_SVM)], axis=1)
subm = subm.rename(columns={0:'target'})
subm.to_csv('sumbission.csv', header=True, index=False)
print("FINISHED")