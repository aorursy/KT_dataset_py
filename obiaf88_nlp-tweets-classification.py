# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importign libraries

import pandas as pd

import matplotlib.pyplot as plt

import re

import nltk

import seaborn as sns

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
#importing data



test = pd.read_csv(r'../input/nlp-getting-started/test.csv')

train = pd.read_csv(r'../input/nlp-getting-started/train.csv')



#data shape

print("Test data has {} rows and {} columns ".format(test.shape[0], test.shape[1]))

print("Train data has {} rows and {} columns ".format(train.shape[0], train.shape[1]))

#check for null values

print("Null for test data :", test.isnull().sum())

print("Null for train data :", train.isnull().sum())



#As we can see keyword and location are the null most of the times (for location about 33% for both test and train data are null) 
#Target variable

print("Distribution of target variable :", train['target'].value_counts())



sns.barplot(x = train['target'].value_counts().index, y = train['target'].value_counts() )

plt.show()
combined_data = [train,test]
#converting all text to lower text



for data in combined_data:

    data['text'] = data['text'].apply(lambda x : x.lower())
# now we can delete both punctuaction and links

def cleaning_punctuaction_and_link(text):

    if pd.isnull(text):

        text = 'na'

    else:

        url = re.compile(r'https?://\S+\www\.\S+')

        text = url.sub('',text)

        punctuaction = "[!-$%&*,.\/:;<=>-?@\^_`{|}~]"

        url2 = re.compile(punctuaction)

        text = url2.sub('',text)

    return text



for data in combined_data:

    data['text'] = data['text'].apply(lambda x : cleaning_punctuaction_and_link(x))

    data['keyword'] = data['keyword'].apply(lambda x: cleaning_punctuaction_and_link(x))

    data['location'] = data['location'].apply(lambda x: cleaning_punctuaction_and_link(x))
tokenizer = nltk.tokenize.WhitespaceTokenizer()



for data in combined_data:

    data['text'] = data['text'].apply(lambda x: tokenizer.tokenize(x))

    data['keyword'] = data['keyword'].apply(lambda x: tokenizer.tokenize(x))

    data['location'] = data['location'].apply(lambda x: tokenizer.tokenize(x))
def removal_stop_words(x):

    if x == 'na':

        words = "na"

    else:

        stop_words = set(stopwords.words('english')) 

        words = [word for word in x if word not in stop_words]

    return words



for data in combined_data:

    data['text'] = data['text'].apply(lambda x: removal_stop_words(x))

    data['keyword'] = data['keyword'].apply(lambda x: removal_stop_words(x))

    data['location'] = data['location'].apply(lambda x: removal_stop_words(x))





def conversion_to_string(x):

    return " ".join(word for word in x)



for data in combined_data:

    data['text'] = data['text'].apply(lambda x : conversion_to_string(x))

    data['keyword'] = data['keyword'].apply(lambda x : conversion_to_string(x))

    data['location'] = data['location'].apply(lambda x : conversion_to_string(x))
stemmer = PorterStemmer()



def stemming_words(x):

    text = [stemmer.stem(word) for word in x.split()]

    return " ".join(text)



 

for data in combined_data:

    data['text'] = data['text'].apply(lambda x: stemming_words(x))

    data['location'] = data['location'].apply(lambda x: stemming_words(x))

    data['keyword'] = data['keyword'].apply(lambda x: stemming_words(x))
def difference_in_words(text1, text2):

    text1_split = []

    for word in text1:

        text1_split.append(word)

    text2_split = []

    for word in text2:

        text2_split.append(word)

    text_difference = set(text1_split).difference(text2_split)

    return text_difference





text_positive = train[train['target'] == 1]['text']

text_negative = train[train['target']== 0]['text']



positive_words = []



for row in text_positive:

    positive_words.append(row)



negative_words = []



for row in text_negative:

    negative_words.append(row)





words_difference = difference_in_words(positive_words, negative_words)
print("Wordcloud for words difference in test positive and negative")

text = " ".join(word for word in words_difference)

wordcloud = WordCloud(max_font_size=50,background_color="white").generate(text)

plt.figure(figsize=[20,10])

plt.axis("off")

plt.imshow( wordcloud, interpolation='bilinear')
# now we can create a final corpus not only with text but also with location and keyword

for data in combined_data:

    data['final_corpus'] = data['text']+ " " + data['location']+ " " + data['keyword']
count_vectorizer = CountVectorizer()



train_vector = count_vectorizer.fit_transform(train['final_corpus']).todense()

test_vector = count_vectorizer.transform(test['final_corpus']).todense()



print("train vector :", train_vector.shape)

print("test vector :", test_vector.shape)
X = train_vector

y = train['target'].values



print("X shape :", X.shape)

print("y shape :", y.shape)



print("test vector shape :", test_vector.shape)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =2020)



logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)



y_pred_lm = logistic_model.predict(X_test)



f1score = f1_score(y_test, y_pred_lm)

print("Logistic Model f1: {}".format(f1score*100))



accuracy_score = accuracy_score(y_test, y_pred_lm)

print("Logistic model accuracy score is {}".format(accuracy_score))
sample_submission = pd.read_csv(r'../input/nlp-getting-started/sample_submission.csv')

sample_submission['target'] = logistic_model.predict(test_vector)

print("Submission head: ", sample_submission.head(10))

sample_submission.to_csv('submission.csv', index=False)