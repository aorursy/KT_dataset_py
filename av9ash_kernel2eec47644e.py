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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.datasets import load_files

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep="\t")

test = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep="\t")

sampleSubmission = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')



train_original = train.copy()

test_original = test.copy()

train.head()

text_train = train["Phrase"]

y_train = train["Sentiment"]
text_test = test["Phrase"]

test.head()
vector = CountVectorizer(analyzer='word', decode_error='ignore', strip_accents='ascii')

print('vectorzing..')

x_train = vector.fit_transform(text_train)

print('done')
clf = RandomForestClassifier(n_jobs=-1, criterion='gini', n_estimators=30, warm_start=True)

print('training..')

clf.fit(x_train, y_train)

print('done')
x_test = vector.transform(text_test)
print('predicting..')

y_preds = clf.predict(x_test)

y_preds
text_test = test["PhraseId"]

print(text_test,y_preds)
with open('output.csv','w') as file:

    file.write('PhraseId,Sentiment \n')

    for i,pred in enumerate(y_preds):

        out = str(text_test[i])+','+str(pred)+'\n'

        print(out)

        file.write(out)



print('done')