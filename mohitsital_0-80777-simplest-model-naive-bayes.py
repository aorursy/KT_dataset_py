# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

import nltk

import string

import re

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

sno= nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

stopwords = set(stopwords.words('english'))



#from sklearn_pandas import DataFrameMapper, cross_val_score

from sklearn.metrics import classification_report

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('Done')
# reading file 

dtrain = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

dtest = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print('Done')

print(dtrain.columns)

print(dtest.columns)
# cleaning "text" column of train and test dataset and saving it to column 'clean_txt'



#stopwords = set(stopwords.word('english'))

# function for removing html

def cleanhtml (sentence):

    cleantext = re.sub(r'http\S+',r'',sentence)

    return cleantext



# function for removing punctuation

def cleanpunc(sentence):

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|)|\|/]',r' ',cleaned)      

    return cleaned

# function for cleaning 'column' of dataframe 'data' and saving cleaned text in column 'clean_txt'

def cleantxt(data,column):

    str1=' '

    final_string=[]

    s=''

    for sent in data[column]:

        filter_sent = []

        rem_html = cleanhtml(sent)

        rem_punc = cleanpunc (rem_html)

        for w in rem_punc.split():

            if ((w.isalpha()) & (len(w)>2)):

                if (w.lower() not in stopwords):

                    s=(sno.stem(w.lower())).encode('utf8')

                    filter_sent.append(s)

                else:

                    continue

            else:

                continue

        str1 = b" ".join(filter_sent)

        final_string.append(str1)

    data['clean_txt'] = np.array(final_string)



cleantxt(dtrain,'text')

cleantxt(dtest,'text')



print(dtrain.columns)

print(dtest.columns)

############ tf-idf ############################

tf_idf_vect = TfidfVectorizer(ngram_range=(1,3)) # one,two and three gram vectorization

tf_idf_mat = tf_idf_vect.fit_transform(dtrain['clean_txt'].values) # fit_transform vectorizer to dtrain['text']

tf_idf_mat_test = tf_idf_vect.transform(dtest['clean_txt'].values) # fit_transform vectorizer to dtest['text']

type(tf_idf_mat)

print(tf_idf_mat.get_shape())

print('done')
# applyig naive bayes , doing cross validation and accuracy matrix

target = dtrain['target']



x, x_test, y, y_test = train_test_split(tf_idf_mat,target,test_size=0.2,train_size=0.8, random_state = 0)



clf = MultinomialNB(alpha=1).fit(x, y)

predicted = clf.predict(x_test)

 

# classification table



def printreport(exp, pred):

    print(pd.crosstab(exp, pred, rownames=['Actual'], colnames=['Predicted']))

    print('\n \n')

    print(classification_report(exp, pred))



printreport(y_test, predicted)
# prediction on dtest data



preds = clf.predict(tf_idf_mat_test)

submission['target']= preds

submission.to_csv('submission.csv',index=False)