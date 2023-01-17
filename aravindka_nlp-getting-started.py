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
import numpy as np
import nltk
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',usecols=['text','target'])
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',usecols=['text'])
wordnet = WordNetLemmatizer()
def clean(item):
    corpus=[]
    for i in range(len(item)):
        comment = re.sub('^[a-zA-Z]',' ',item[i])
        comment = comment.lower()
        comment = comment.split()
        comment = [wordnet.lemmatize(word) for word in comment if not word in stopwords.words('english')]
        comment = ' '.join(comment)
        corpus.append(comment)
    return corpus
train_clean = clean(train['text'])
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
train_data = cv.fit_transform(train_clean).toarray()
train_data
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(train_data,train['target'])
pred = classifier.predict(train_data)
score = accuracy_score(train['target'],pred)
score
test_clean = clean(test['text'])
test_data = cv.transform(test_clean).toarray()
test_data
val = classifier.predict(test_data)
# val = pd.DataFrame(val)
val
sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample['target'] = val
sample.head()
sample.to_csv('submission.csv',index=False)