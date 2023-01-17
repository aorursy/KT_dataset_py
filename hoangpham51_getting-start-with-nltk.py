# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import string

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score # for evaluating results
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
# Load stop word
eng_stopwords = set(stopwords.words("english"))
def processingData(train, test):
    # Tokenize
    train['question_text'] = train["question_text"].apply(lambda x: " ".join(word_tokenize(str(x))))
    test['question_text'] = test["question_text"].apply(lambda x: " ".join(word_tokenize(str(x))))

    # Remove punctuation
    train['question_text'] = train["question_text"].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    test['question_text'] = test["question_text"].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

    ## Remove stopwords in the text ##
    train["question_text"] = train["question_text"].apply(lambda x: " ".join([w for w in str(x).lower().split() if not w in eng_stopwords]))
    test["question_text"] = test["question_text"].apply(lambda x: " ".join([w for w in str(x).lower().split() if not w in eng_stopwords]))
    
    return train, test
train, test = processingData(train, test)
train.head()
test.head()
train_question_list = train['question_text']
test_question_list = test['question_text']

vectorizer  = CountVectorizer()

x_train =  vectorizer.fit_transform(train_question_list)
x_test =  vectorizer.transform(test_question_list)
y_train_tfidf = np.array(train["target"].tolist())
train_x, validate_x, train_y, validate_y = train_test_split(x_train, y_train_tfidf, test_size=0.3)
clf = MultinomialNB()
clf.fit(train_x, train_y)
y_vad = clf.predict(validate_x)
print('accuracy = %.2f%%' % \
      (accuracy_score(validate_y, y_vad)*100))
y_predict = clf.predict(x_test)
predict = pd.DataFrame(data = y_predict, columns=['prediction'])
predict = predict.astype(int)
id = test['qid']
id_df = pd.DataFrame(id)
# Join predicted into result dataframe and write result as a CSV file
result = id_df.join(predict)
result.to_csv("submission.csv", index = False)