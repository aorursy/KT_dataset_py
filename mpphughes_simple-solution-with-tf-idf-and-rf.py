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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
###### This is a simple solution using tf idf as features and a Random Forest classifier to assign sentiment class. Initially I tried this with Word2Vec 
###### features but ultimately I found that using tf idf was way more accurate!
train = pd.read_csv('../input/train.tsv',sep='\t')
test = pd.read_csv('../input/test.tsv',sep='\t')
train.head()
test.head()
text = train['Phrase']
test_text = test['Phrase']
tfidf=TfidfVectorizer(ngram_range=(1,2),max_df=0.95,min_df=10,sublinear_tf=True)
trainv =tfidf.fit_transform(text)
testv = tfidf.transform(test_text)
trainv.shape
testv.shape
y = train['Sentiment']
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(trainv, y)
y_pred = clf.predict(testv)

sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
sub['Sentiment'] = y_pred
sub.to_csv("submission.csv", index=False)