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

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn import svm
train_data = pd.read_csv('../input/train.tsv',delimiter='\t',encoding='utf-8')

train_data = train_data.sample(frac=0.01)
train_data.head()
abc = TfidfVectorizer()

train_vectors = abc.fit_transform(train_data['Phrase'])

train_labels = train_data['Sentiment']
model = svm.SVC(kernel='linear', class_weight="balanced")

model.fit(train_vectors, train_labels)
test_data = pd.read_csv('../input/test.tsv',delimiter='\t',encoding='utf-8') 

test_data.head()
columns = ['PhraseId', 'Sentiment']

submission_data = pd.DataFrame(columns=columns)

for index in range(len(test_data)):

  input_text = test_data.iloc[index]['Phrase']

  test_vector = abc.transform([input_text])

  x = model.predict(test_vector)

  submission_data.loc[len(submission_data)] = [test_data.iloc[index]['PhraseId'], x]

submission_data.head()
submission_data['Sentiment'] = [x[0] for x in submission_data['Sentiment']]
submission_data.to_csv('submission.csv')