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
df = pd.read_csv('../input/train.csv')
df.head()
import textblob
del df['id']

train = df.as_matrix()
##from textblob.classifiers import NaiveBayesClassifier
##clf = NaiveBayesClassifier(train[0:5000])
##clf.accuracy(train[5000:6000])
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction import text
st_wd = text.ENGLISH_STOP_WORDS

classifier = Pipeline([('vect', CountVectorizer(stop_words=st_wd,lowercase=1)),

                      ('clf', MLPClassifier(hidden_layer_sizes=(10),solver='sgd',max_iter=50,random_state=20,verbose=10,learning_rate_init=.07)),])

##c_vector = CountVectorizer()

##X_train_counts = c_vector.fit_transform(df['text'].values)
X_train = df['text']

targets = df['author'].values

classifier.fit(X_train, targets)

classifier.score(X_train,targets)
##tx = ['A youth passed in solitude, my best years spent under your gentle and feminine fosterage, has so refined the groundwork of my character that I cannot overcome an intense distaste to the usual brutality exercised on board ship: I have never believed it to be necessary, and when I heard of a mariner equally noted for his kindliness of heart and the respect and obedience paid to him by his crew, I felt myself peculiarly fortunate in being able to secure his services.']

##classifier.predict(c_vector.transform(tx))
##classifier.score(X_train_counts,targets)

df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test.head()
##X_test = c_vector.transform(df_test['text'].values)

X_test = df_test['text']
Y_test = classifier.predict_proba(X_test)
Y_test
sub = pd.read_csv('../input/sample_submission.csv')
sub['EAP'] = [ '{:f}'.format(x) for x in Y_test[:,0]]

sub['HPL'] = [ '{:f}'.format(x) for x in Y_test[:,1]]

sub['MWS'] = [ '{:f}'.format(x) for x in Y_test[:,2]]
sub.head()
sub.to_csv('output.csv',index=False)