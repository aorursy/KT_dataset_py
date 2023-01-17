# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv',usecols=['title','description'])
data.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re

import pickle 
#import mglearn
import time


from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
data.description.values[:10]
txt = ['Кокон для сна малыша,пользовались меньше месяца.цвет серый',
       'Стойка для одежды, под вешалки. С бутика.',
       'В хорошем состоянии, домашний кинотеатр с blu ray, USB. Если настроить, то работает смарт тв /\nТорг',
       'Продам кресло от0-25кг', 'Все вопросы по телефону.',
       'В хорошем состоянии',
       'Электро водонагреватель накопительный на 100 литров Термекс ID 100V, плоский, внутренний бак из нержавейки, 2 кВт, б/у 2 недели, на гарантии.',
       'Бойфренды в хорошем состоянии.', '54 раз мер очень удобное',
       'По стельке 15.5см мерить приокский район. Цвет темнее чем на фото']
# Initialize a CountVectorizer object: count_vectorizer
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)

# Transforms the data into a bag of words
count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

# Print the first 10 features of the count_vec
print("Every feature:\n{}".format(count_vec.get_feature_names()))
print("\nEvery 3rd feature:\n{}".format(count_vec.get_feature_names()[::3]))
print("Vocabulary size: {}".format(len(count_train.vocabulary_)))
print("Vocabulary content:\n {}".format(count_train.vocabulary_))
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 3), max_df=1.0, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 1), max_df=1.0, min_df=0.3, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
print("\nOnly 'park' becomes the vocabulary of the document term matrix (dtm) because it appears in 2 out of 3 documents, \
meaning 0.66% of the time.\
      \nThe rest of the words such as 'big' appear only in 1 out of 3 documents, meaning 0.33%. which is why they don't appear")
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 1), max_df=0.50, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
print("\nOnly 'park' is ignored because it appears in 2 out of 3 documents, meaning 0.66% of the time.")
count_vec = CountVectorizer(stop_words=stopwords.words('russian'), analyzer='word', encoding='KOI8-R',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=4)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
data.description.values[10:30]
txt1 = ['Семейная пара из двух человек снимет коттедж с дизайнерским ремонтом, со стильной мебелью и бытовой техникой. Гараж на 2 машины, 2 санузла. Огороженная территория, сигнализация.  Рассмотрим варианты коттеджей, которые выставлены на продажу. Ежемесячная оплата до 100 тыс.руб. в месяц.  Агентства просьба не беспокоить.',
       'Дом находиться внутри квартала./\nПластиковые окна во всей квартире, балкон застеклен железом, обшит деревом, в комнате натяжной потолок, новый линолиум,/\nванна и туалет- стеновые панели.Квартира подходит под ипотеку, 1 собственник, ЧП, небольшой торг возможен.']
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word',
                     stop_words=stopwords.words('russian'), encoding='KOI8-R')
txt_fitted = tf.fit(txt1)
txt_transformed = txt_fitted.transform(txt1)
print ("The text: ", txt1)
print ("The txt_transformed: ", txt_transformed)
tf.vocabulary_
idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))
print("\nWe see that the tokens 'sang','she' have the most idf weight because \
they are the only tokens that appear in one document only.")
print("\nThe token 'not' appears 6 times but it is also in all documents, so its idf is the lowest")
rr = dict(zip(txt_fitted.get_feature_names(), idf))
token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
token_weight.columns=('token','weight')
token_weight = token_weight.sort_values(by='weight', ascending=False)[:10]
token_weight

sns.barplot(x='token', y='weight', data=token_weight)
plt.title("Inverse Document Frequency(idf) per token")
fig=plt.gcf()
fig.set_size_inches(10,5)
fig.set_figwidth(10)
plt.show()
# get feature names
feature_names = np.array(tf.get_feature_names())
sorted_by_idf = np.argsort(tf.idf_)
print("Features with lowest idf:\n{}".format(
       feature_names[sorted_by_idf[:3]]))
print("\nFeatures with highest idf:\n{}".format(
       feature_names[sorted_by_idf[-3:]]))
data.description.values[132]
print("The token 'not' has  the largest weight in document #2 because it appears 3 times there. But in document #1\
 its weight is 0 because it does not appear there.")
txt_transformed.toarray()
new1 = tf.transform(txt1)

# find maximum value for each of the features over all of dataset:
max_val = new1.max(axis=0).toarray().ravel()

#sort weights from smallest to biggest and extract their indices 
sort_by_tfidf = max_val.argsort()

print("Features with lowest tfidf:\n{}".format(
      feature_names[sort_by_tfidf[:3]]))

print("\nFeatures with highest tfidf: \n{}".format(
      feature_names[sort_by_tfidf[-3:]]))