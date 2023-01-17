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
#importing the libraries
import pandas as pd
import numpy as np
news_train_test = pd.read_csv('/kaggle/input/odia-news-dataset/train.csv')
news_validate = pd.read_csv('/kaggle/input/odia-news-dataset/valid.csv')
print(news_train_test.head(5))
#importing the stopword dictionary. You can create your own dictionary.
stopwords = pd.read_csv('/kaggle/input/stopwords-odia/stopwords.csv')
sw_arr = stopwords.to_numpy()
print(sw_arr)
def removePunctuations(headline):
    headline = headline.replace(',',' ')
    headline = headline.replace(':',' ')
    headline = headline.replace(';',' ')
    headline = headline.replace('.',' ')
    headline = headline.replace('\'','')
    headline = headline.replace('-',' ')
    return headline;
news_arr = []
for headline in news_train_test['headings'] :
    filtered_news_string = ''
    headline=removePunctuations(headline)
    for word in headline.split(' '):
        if word not in sw_arr:
            filtered_news_string = filtered_news_string+word+' '
    news_arr.append(filtered_news_string)
#creating a new dataframe to store the filtered news array headlines
dataset_new = pd.DataFrame(news_arr, columns=['filter_news'])
print(dataset_new.head(5))

#Now we will concat the filtered news dataset with our original news corpus so that we can get the labels against filtered headlines
news_train_test = pd.concat([news_train_test, dataset_new], axis = 1)
print(news_train_test.columns)
#importing CountVectorizer to create vectors
from sklearn.feature_extraction.text import CountVectorizer

#we will vectorize each words in a documents.these vectors will be our features to train the model
vectorizer = CountVectorizer(analyzer = "word",max_features = 1700)
x= vectorizer.fit_transform(news_train_test['filter_news']).toarray()

#now we will store our features in x
print(x)
#our target variable
print(news_train_test['label'].unique)
#importing library
from sklearn.preprocessing import LabelEncoder
cat_encoder = LabelEncoder()

#encoding the label column from our news corpus as type category
y = cat_encoder.fit_transform(news_train_test['label'].astype('category'))
print(y)
#importing the library
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('Shape of train data')
print(x_train.shape)
print(y_train.shape)

print('Shape of test data')
print(x_test.shape)
print(y_test.shape)
#Model building Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

#fitting our train data with our classifier to create the model
classifier.fit(x_train, y_train)

print('Training data accuracy')
print(classifier.score(x_train , y_train))
y_pred = classifier.predict(x_test)
print('Test data accuracy')
print(classifier.score(x_test , y_test))