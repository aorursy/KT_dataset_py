# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



import re, string 

def processTweet(tweet):

    # Remove HTML special entities (e.g. &amp;)

    tweet = re.sub(r'\&\w*;', '', tweet)

    #Convert @username to AT_USER

    tweet = re.sub('@[^\s]+','',tweet)

    # Remove tickers

    tweet = re.sub(r'\$\w*', '', tweet)

    # To lowercase

    tweet = tweet.lower()

    # Remove hyperlinks

    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)

    # Remove hashtags

    tweet = re.sub(r'#\w*', '', tweet)

    # Remove Punctuation and split 's, 't, 've with a space for filter

    tweet = re.sub(r'[' + string.punctuation + ']+', ' ', tweet)

    # Remove words with 2 or fewer letters

    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)

    # Remove whitespace (including new line characters)

    tweet = re.sub(r'\s\s+', ' ', tweet)

    # Remove single space remaining at the front of the tweet.

    tweet = tweet.lstrip(' ') 

    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:

    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 

    return tweet
# clean dataframe's text column

train['text'] = train['text'].apply(processTweet)

test['text'] = test['text'].apply(processTweet)



train[train['target']==1]
X_train=train.text

y_train=train.target

X_test=test.text

from sklearn.pipeline import Pipeline                         #Pipe

#from sklearn.naive_bayes import MultinomialNB                # Training Naive Bayes (NB) classifier on training data.

from sklearn.feature_extraction.text import TfidfTransformer  #TF-IDF

from sklearn.feature_extraction.text import CountVectorizer   # Extracting features from text files

from lightgbm import LGBMClassifier



pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2)))

                     , ('tfidf', TfidfTransformer())

                     , ('LGBMmodel', LGBMClassifier(n_estimators=2000,learning_rate=0.05))])



#parameters = {'vect__ngram_range': [(1, 1), (1, 2)],

#             'tfidf__use_idf': (True, False),

#              'LGBMmodel__alpha': (1e-2, 1e-3),

#             }
# do 10-fold cross validation for each of the 8 possible combinations of the above params

#from sklearn.model_selection import GridSearchCV

#grid = GridSearchCV(pipe, cv=10, param_grid=parameters, verbose=1)
#grid.fit(X_train, y_train)

# summarize results

#print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))

#print('\n')

#means = grid.cv_results_['mean_test_score']

#stds = grid.cv_results_['std_test_score']

#params = grid.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):

#    print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))
pipe.fit(X_train,y_train)
train_pred=pipe.predict(X_train)

from sklearn import metrics

metrics.confusion_matrix(y_train,train_pred)
print(metrics.classification_report(y_train, train_pred, digits=3))
#X_test=X_test.as_matrix()



test_preds=pipe.predict(X_test)

output = pd.DataFrame({'id': test.id,

                       'target': test_preds})

output.to_csv('submission0.csv', index=False)