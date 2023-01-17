# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')

train.head()
test.head()
subm.head()
train.shape
text_length = train.text.str.len()

text_length.mean(), text_length.max()

#Groupby is a very useful function, it group by items present in author column

train.groupby('author').size()

from sklearn import preprocessing
#Using LabelEncoder we are converting the non_numerical label into numerical label.

#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

le = preprocessing.LabelEncoder()

le.fit(train.author)

num_of_labels = len(list(le.classes_))

list(le.classes_)

y_labels = le.transform(train.author) 

y_labels
le.inverse_transform(y_labels)
#this returns row and column array of all the empty values present in our dataframe

row, column = np.where(pd.isnull(train))

print (row, column)
#Following regularexpression and tokenizer is based on Jeremy Howard kernel.

#https://www.kaggle.com/jhoward

import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
TEXT = 'text'

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_term_doc = vec.fit_transform(train[TEXT])

test_term_doc = vec.transform(test[TEXT])
trn_term_doc, test_term_doc
x = trn_term_doc

test_x = test_term_doc
from sklearn.linear_model import LogisticRegression



def get_model(y):

    m = LogisticRegression(C=4, dual=True)

    return m.fit(x, y) 
preds = np.zeros((len(test), num_of_labels))

m = get_model(y_labels)

preds = m.predict_proba(test_x)
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = ['EAP','HPL','MWS'])], axis=1)

submission.to_csv('submission.csv', index=False)

submission.head()