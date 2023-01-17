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
TRAIN_FILE= '/kaggle/input/nlp-getting-started/train.csv'

TEST_FILE= '/kaggle/input/nlp-getting-started/test.csv'

SUBMISSION_FILE= '/kaggle/input/nlp-getting-started/sample_submission.csv'
df_train=pd.read_csv(TRAIN_FILE)

df_test=pd.read_csv(TEST_FILE)

df_submission=pd.read_csv(SUBMISSION_FILE)
#Balanced classes

df_train['target'].value_counts()


from nltk.corpus import stopwords

import nltk

import string

import re

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

sno= nltk.stem.SnowballStemmer('english')

stopwords = set(stopwords.words('english'))
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



cleantxt(df_train,'text')

cleantxt(df_test,'text')
X=df_train['clean_txt']

y=df_train['target']



Xtest=df_test['clean_txt']
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer=TfidfVectorizer(ngram_range=(1,3))

Xtrain_trans=vectorizer.fit_transform(X)

Xtest_trans=vectorizer.transform(Xtest)
import sklearn

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(Xtrain_trans,y,random_state=0)

from sklearn.naive_bayes import MultinomialNB



clf=MultinomialNB(alpha=1).fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred=clf.predict(Xtest_trans)
y_pred
df_submission['target'] = y_pred
df_submission.to_csv('submission.csv', index=False)