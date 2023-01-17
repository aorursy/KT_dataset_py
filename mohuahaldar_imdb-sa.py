# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tr_data=pd.read_csv('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv')

tr_data.head()
tr_data.shape
import nltk

nltk.download('punkt')

import string

from nltk import word_tokenize

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
class text_preprocessing:

  def __init__(self):

    pass

  # Convert text into lower case

  def convert_to_lower(self, text):

    return text.lower()



  # Remove punctuation from text

  def remove_punctuation(self, text):



    return "".join([c for c in text if c not in string.punctuation])

  # Tokenize text

  def tokenize_text(self, text):

    words=word_tokenize(text)

    return words

  # Remove stopwords

  def remove_stopwords(self, tokens):

    stopwrds=stopwords.words('english')

    token_clean=[token for token in tokens if token not in stopwrds and len(token)>3]

    return token_clean

  # stem the tokens

  def stem_tokens(self, tokens):

   stemmed_tokens=[PorterStemmer().stem(token) for token in tokens]

   return stemmed_tokens

  
tp=text_preprocessing()

lower=tr_data.iloc[:,:1].apply(lambda row:tp.convert_to_lower(row.str), axis=1)

sen_cleaned=lower.apply(lambda row:tp.remove_punctuation(row.values), axis=1)

#tokens=sen_cleaned.apply(lambda row:tp.tokenize_text(row))

#cleaned_tokens=tokens.apply(lambda row:tp.remove_stopwords(row))

#stemmed_tokens=cleaned_tokens.apply(lambda row:tp.stem_tokens(row))
tst_data=pd.read_csv('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv')

tst_data.shape
# preprocess the test data

lower=tst_data.iloc[:,:1].apply(lambda row:tp.convert_to_lower(row.str), axis=1)

sen_cleaned_tst=lower.apply(lambda row:tp.remove_punctuation(row.values), axis=1)

full_text_df=pd.concat([sen_cleaned,sen_cleaned_tst])

full_text_df.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer( analyzer='word',ngram_range=(1,1),sublinear_tf=True)
full_vocab=tv.fit(full_text_df)

tr_X=tv.transform(sen_cleaned)

y_train=tr_data['label']

y_train.head()
from sklearn.svm import LinearSVC

SVM=LinearSVC()

SVM.fit(tr_X,y_train.values)
y_tst=tst_data['label']

test_X=tv.transform(sen_cleaned_tst)

y_tst_pred=SVM.predict(test_X)
from sklearn.metrics import accuracy_score
accuracy_score(y_tst, y_tst_pred)