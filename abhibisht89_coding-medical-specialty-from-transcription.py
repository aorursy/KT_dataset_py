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
data=pd.read_csv('../input/medicaltranscriptions/mtsamples.csv')
data.head()
data.transcription.isna().sum()
data.medical_specialty.isna().sum()
data.shape
filter_df=data[["transcription","medical_specialty"]].dropna()
filter_df.shape
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from time import time
def get_vectoriser(type='cv', ngram_range=(1, 1)):
    if type == 'tfidf':
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern = r"(?u)\b\w+\b",
                                    ngram_range=ngram_range
                                    )
    
    else: 
    #     by default go to count vectoriser
        # add token_pattern to handle 1 char words also.
        vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b", ngram_range=ngram_range)
    
    return vectorizer
tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000],
                     'class_weight': ['balanced', None]
                    },
                    {'kernel': ['linear'], 
                     'C': [1, 10, 100, 1000],
                     'class_weight': ['balanced', None]
                    }
                   ]
clf = GridSearchCV(SVC(probability=True), tuned_parameters, n_jobs=8) 
vectorizer = get_vectoriser(type='tfidf')
X_train = vectorizer.fit_transform(filter_df.transcription)
print("n_samples: %d, n_features: %d" % X_train.shape)
feature_names = vectorizer.get_feature_names()
import numpy as np
feature_names = np.asarray(feature_names)
y_train=filter_df.medical_specialty
# clf.fit(X_train, y_train)
import pickle
# save the model to disk
filename = r'../input/coding-medical-specialty-from-transcription/model.pkl'
# pickle.dump(clf, open(filename, 'wb')) 

# load the model from disk
clf = pickle.load(open(filename, 'rb'))
y_train_pred = clf.predict(X_train)
accuracy_score = metrics.accuracy_score(y_train, y_train_pred)
print("accuracy:   %0.3f" % accuracy_score)
