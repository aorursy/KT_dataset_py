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

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_data.head()

from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

def cleanText(text):
    words = regexp_tokenize(text.lower(), r'[A-Za-z]+')
    words = [wordnet_lemmatizer.lemmatize(w) for w in words if w not in stopwords and len(w)>2]
    cleaned_text = ' '.join(words)
    return cleaned_text

train_data['cleaned_text'] = train_data['text'].map(lambda x:cleanText(x))
test_data['cleaned_text'] = test_data['text'].map(lambda x:cleanText(x))

#train_data.head()
#test_data.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.90, min_df=5, stop_words='english')
tfidf.fit(train_data['cleaned_text'])
X_train = tfidf.transform(train_data['cleaned_text'])
X_test = tfidf.transform(test_data['cleaned_text'])
y_train = train_data['target'].values
num_classes = np.max(y_train) + 1

from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(random_state=95, objective='binary:logistic', n_estimators=100, max_depth=None, num_classes = num_classes)
xgb_classifier.fit(X_train, y_train)

y_predict = xgb_classifier.predict(X_test)

resultant_corpus = pd.concat([test_data['id'], pd.DataFrame(y_predict, columns=['target'])], axis=1)
#resultant_corpus.head()

resultant_corpus.to_csv('Raj_Ankit_submission.csv')
