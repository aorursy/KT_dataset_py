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
import numpy as np

import pandas as pd

%matplotlib inline



import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/fake-news/train.csv')

test = pd.read_csv('../input/fake-news/test.csv')

train.shape ,test.shape
train.sample(4)
train.isna().sum()
# Percentage of missing value columns

train.isna().sum()/len(train)*100
x = train.drop('label' ,axis =1)

y = train['label']

x.head(3)
train = train.dropna()

train.shape
text_df = train.copy()

text_df.reset_index(inplace =True)

text_df.head(5)
text_df['title'][6]
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0,len(text_df)):

    review = re.sub('[^a-zA-Z]' ,' ',text_df['title'][i])

    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)

#Applying Vectorizer for text Processing

from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer ,HashingVectorizer

cv = CountVectorizer(max_features= 5000 , ngram_range= (1,3))

X = cv.fit_transform(corpus).toarray()

X.shape
cv.get_feature_names()
y = text_df['label']

y.shape
from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train ,y_test =train_test_split(X,y ,test_size = .30 , random_state = 101)

x_train.shape ,x_test.shape ,y_train.shape ,y_test.shape
# randomforest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train ,y_train)

pred = rf.predict(x_test)

pred
from sklearn.metrics import accuracy_score

print("Accuracy   =",accuracy_score(y_test ,pred))
test.sample()
test.isna().sum()
test.dtypes
# Handling missing value

for i in test.columns:

    test [i] = test[i].fillna(test[i].mode()[0])
test.shape

test.reset_index(inplace =True)
corpus_test = []

for i in range(0 ,len(test)):

    a = re.sub('[^a-zA-Z]' ,' ' , test['title'][i])

    a = a.lower()

    a = a.split()

    

    a = [ps.stem(word) for word in a if not word in stopwords.words('english')]

    a = ' '.join(a)

    corpus_test.append(a)

corpus_test
#Vectorizer

tst = cv.fit_transform(corpus_test).toarray()

tst.shape
# Accuracy for TEST

p = rf.predict(tst)

p
submission = pd.read_csv('../input/fake-news/submit.csv')

submission.shape
submission.head(3)
predictin = pd.DataFrame(p)

predictin ['id'] = submission['id']

predictin['label'] = predictin[0]

predictin = predictin.drop(0 ,axis =1)

predictin
my_submission = predictin.to_csv()