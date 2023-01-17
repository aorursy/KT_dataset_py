# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, f1_score





quora = pd.read_csv('../input/train.csv', nrows=100000)

test = pd.read_csv('../input/test.csv')
quora.shape
import string

from nltk.corpus import stopwords



#Tokenization function

def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
nb_pipeline = Pipeline([

    ('vectorizer', CountVectorizer(analyzer=text_process)),

    ('tfidf_transformer', TfidfTransformer())

])
k_fold = KFold(n_splits = 3)

nb_f1_scores = []

nb_conf_mat = np.array([[0, 0], [0, 0]])



for train_indices, test_indices in k_fold.split(quora):

    

    train_text = quora.iloc[train_indices]['question_text'].values

    train_y = quora.iloc[train_indices]['target'].values



    test_text = quora.iloc[test_indices]['question_text'].values

    test_y = quora.iloc[test_indices]['target'].values



    vectorized_text = nb_pipeline.fit_transform(train_text)



    sm = SMOTE(sampling_strategy=0.2,random_state=42,n_jobs=-1)

    train_text_res, train_y_res = sm.fit_sample(vectorized_text, train_y)



    clf = MultinomialNB()

    clf.fit(train_text_res, train_y_res)

    predictions = clf.predict(nb_pipeline.transform(test_text))

    

    nb_conf_mat += confusion_matrix(test_y, predictions)

    score1 = f1_score(test_y, predictions)

    nb_f1_scores.append(score1)



print("F1 Score: ", sum(nb_f1_scores)/len(nb_f1_scores))

print("Confusion Matrix: ")

print(nb_conf_mat)
