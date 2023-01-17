import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

from sklearn import feature_extraction, linear_model, model_selection, preprocessing, neighbors,ensemble, naive_bayes,gaussian_process

import xgboost as xgb

import matplotlib.pyplot as plt 

import scipy



from sklearn.naive_bayes import MultinomialNB



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#download the data and store in variables

data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")





print(data)
#check to see the proportion of positives and negatives in the data

print(data[data.target == 1].target.count())

print(data[data.target == 0].target.count())
def tweet_length_matrix(Series):

    """

    This function takes a series and returns a scipy sparse matrix of 2 dimensions of the tweet

    """

    len_tweet = Series.map(len)

    len_squared = Series.map(lambda x : len(x)**2)

    

    sparse_len = scipy.sparse.csr_matrix(pd.concat([len_tweet,len_squared],axis = 1))

    

    return sparse_len
#actually use model

#define the count vectorizer based on ridge regression

count_vectorizer = feature_extraction.text.CountVectorizer(ngram_range=(1,1))

#min_df=0.002

#ngram_range=(1,1),min_df=0.008

train_vectors = count_vectorizer.fit_transform(data["text"])

test_vectors = count_vectorizer.transform(test["text"])



#create a length matrix with the length and length squared of each tweet for the training and test sets

len_train = tweet_length_matrix(data["text"])

len_test = tweet_length_matrix(test["text"])

    

train_vectors= scipy.sparse.hstack([train_vectors,len_train])

test_vectors= scipy.sparse.hstack([test_vectors,len_test])



#define ridge classifier model

nb = MultinomialNB()

#fit ridge classifier

nb.fit(train_vectors, data["target"])



#make and submit predictions

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = nb.predict(test_vectors)

sample_submission.head()

sample_submission.to_csv("submission.csv", index=False)