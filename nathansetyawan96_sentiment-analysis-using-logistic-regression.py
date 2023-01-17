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
import pandas as pd



amazon_file = '../input/amazon_cells_labelled.txt'

df = pd.read_csv(amazon_file, sep='\t', comment='#', na_values='Nothing', names=['comments', '0/1'], header = None)
pd.isnull(df).any()
df['0/1'].unique()
df['0/1'].value_counts()
df.info()
df['comments'] = df.apply(lambda row: str(row['comments']).lower(), axis=1)
df.head()
from string import punctuation

def remove_punctuations(string):

    return ''.join(c for c in string if c not in punctuation)
df['comments'] = df.apply(lambda row: remove_punctuations(row['comments']), axis=1)
df.head()
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))



def remove_stopwords(string):    

    tokenized = word_tokenize(string)

    filtered_sentence = [word for word in tokenized if not word in stop_words]

    return ' '.join(c for c in filtered_sentence)
df['comments'] = df.apply(lambda row: remove_stopwords(row['comments']), axis=1)
df.head()
def convert(integer):

    if(integer == 1):

        return 'Positive'

    else:

        return 'Negative'
df['0/1'] = df.apply(lambda row: convert(row['0/1']), axis=1)
from sklearn.model_selection import train_test_split



X = df['comments']

y = df['0/1']



#
one_hot_encoded_label = pd.get_dummies(y)

one_hot_encoded_label.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(min_df=2, ngram_range=(1, 1))

X_train = vect.fit(X_train).transform(X_train) 

X_test = vect.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



c_val = [0.75, 1, 2, 3, 4, 5, 10]



for c in c_val:

    logreg = LogisticRegression(C=c)

    logreg.fit(X_train, y_train)

    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, logreg.predict(X_test))))