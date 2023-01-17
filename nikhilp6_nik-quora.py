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
import nltk
data = pd.read_csv("../input/quora-dataset/train.csv")
data_test = pd.read_csv("../input/quora-dataset/test.csv")
submission = pd.read_csv("../input/quora-dataset/sample_submission.csv")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def clean(text):
    wn = nltk.WordNetLemmatizer()
    stopword = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    lower = [word.lower() for word in tokens]
    no_stopwords = [word for word in lower if word not in stopword]
    no_alpha = [word for word in no_stopwords if word.isalpha()]
    lemm_text = [wn.lemmatize(word) for word in no_alpha]
    clean_text = lemm_text
    return clean_text
data['clean']=data['question_text'].map(clean)
data['clean_text']=data['clean'].apply(lambda x: " ".join([str(word) for word in x]))

#Test data
data_test['clean']=data_test['question_text'].map(clean)
data_test['clean_text']=data_test['clean'].apply(lambda x: " ".join([str(word) for word in x]))
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()

from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf = TfidfVectorizer()
#cv_data = cv.fit_transform(data['clean_text'])
#tfidf_data = tfidf.fit_transform(data['clean_text'])
#tfidf_test = tfidf.fit_transform(data_test['clean_text'])

tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
a=tfidf.fit_transform(data['question_text'].values.tolist() + data_test['question_text'].values.tolist())
tfidf_data = tfidf.transform(data['question_text'].values.tolist())
tfidf_test = tfidf.transform(data_test['question_text'].values.tolist())

train_tfidf = tfidf.transform(data['question_text'].values.tolist())
test_tfidf = tfidf.transform(data_test['question_text'].values.tolist())
X = train_tfidf
y = data['target']
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3,shuffle=False)
X_train = X
y_train = y
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(max_iter=1400000).fit(X_train,y_train)
y_pred = regressor.predict(test_tfidf)
submission['prediction'] = y_pred
submission.to_csv("submission.csv", index=False)