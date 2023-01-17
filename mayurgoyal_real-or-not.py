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
import pandas as pd

import re

import numpy as np

import nltk

nltk.download('wordnet')

from nltk import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def remove_special(text,pattern):

    r=re.findall(pattern,text)

    for i in r:

        text=re.sub(i,'',text)

    return text
data['text'] = np.vectorize(remove_special)(data['text'],'@[\w]*')

data['text'] = np.vectorize(remove_special)(data['text'],'#[\w]*')

data['text'] = np.vectorize(remove_special)(data['text'],'[1-9]')

data['text']=data['text'].str.replace('[^a-zA-Z#]',' ')

data['text'] = data['text'].apply(lambda x: ' '.join([i for i in x.split() if len(i) > 3]))
lm=WordNetLemmatizer()

tweets=data['text'].apply(lambda x: x.split())

tweets=tweets.apply(lambda x: [lm.lemmatize(i) for i in x if i not in stopwords.words('english')])
for i in range(len(tweets)):

    tweets[i]=' '.join(tweets[i])

data['text']=tweets
text=data['text']

cv=CountVectorizer(max_features=5000,ngram_range=(1,3))

X=cv.fit_transform(text).toarray()
X
y=data['target']

y=y.astype('int')
classifier=MultinomialNB()

classifier.fit(X,y)
test_data=test['text']
rev=cv.transform(test_data)

val=classifier.predict(rev)
val=pd.DataFrame(val)

val.to_csv('new_test.csv',index=False)