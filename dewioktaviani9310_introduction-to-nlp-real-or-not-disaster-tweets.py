# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import naive_bayes, svm

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print(train_df.shape)

print(test_df.shape)
print(train_df.info())

print(test_df.info())
print(train_df.isna().sum())

print(test_df.isna().sum())
train_df["location"] = train_df["location"].fillna("unknown")

test_df["location"] = test_df["location"].fillna("unknown")

print(train_df.isna().sum())

print(test_df.isna().sum())
train_df[train_df["target"] == 0]["text"].iloc[0]
train_df[train_df["target"] == 1]["text"].iloc[0]
def length(text):    

    '''a function which returns the length of text'''

    return len(text)
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

example_train_vectors
print(example_train_vectors[2].todense().shape)

print(example_train_vectors[2].todense())
# train data

train_vectors = count_vectorizer.transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
train_vectors.data
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
train_df['text'] = [entry.lower() for entry in train_df['text']]
train_df['text']= [word_tokenize(entry) for entry in train_df['text']]
train_df['text']
tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV
for index,entry in enumerate(train_df['text']):

    # Declaring Empty List to store the words that follow the rules for this step

    Final_words = []

    # Initializing WordNetLemmatizer()

    word_Lemmatized = WordNetLemmatizer()

    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only alphabets

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words.append(word_Final)

    # The final processed set of words for each iteration will be stored in 'text_final'

    train_df.loc[index,'text_final'] = str(Final_words)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train_df['text_final'],train_df['target'],test_size=0.3)
Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(train_df['text_final'])



Train_X_Tfidf = Tfidf_vect.transform(Train_X)

Test_X_Tfidf = Tfidf_vect.transform(Test_X)
Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset

predictions_NB = Naive.predict(Test_X_Tfidf)



# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# Classifier - Algorithm - SVM

# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(Train_X_Tfidf,Train_Y)



# predict the labels on validation dataset

predictions_SVM = SVM.predict(Test_X_Tfidf)



# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)