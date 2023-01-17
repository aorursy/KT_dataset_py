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
df = pd.read_csv('../input/labeledTrainData.tsv', sep='\t',escapechar='\\')
df["review"].head()
from bs4 import BeautifulSoup

def clean_text(text):

    text = BeautifulSoup(text, 'html.parser').get_text()

    return text
df["clean_review"] = df.review.apply(clean_text)

df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction import text

tf_vec = TfidfVectorizer(lowercase=1,min_df=0.001,stop_words=text.ENGLISH_STOP_WORDS)

X_voc = df["clean_review"].values

tf_vec.fit(X_voc)
X_train = df["clean_review"][0:20000].values  

X_tf =  tf_vec.transform(X_train)
X_tf.shape
clf = MultinomialNB()

target = df["sentiment"][0:20000].values

clf.fit(X_tf,target)

X_test = df["clean_review"][20000:25000].values.astype("U")

test_x_tf = tf_vec.transform(X_test)

Y_test = df["sentiment"][20000:25000]

clf.score(test_x_tf,Y_test)