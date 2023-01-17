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
import numpy as np
train = pd.read_csv('../input/train.csv')
train.head()
train['target'].value_counts() / train.shape[0] * 100  # imbalanced data
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud().generate('i love india; i love bikes; i love apples')
plt.imshow(wc)
wc = WordCloud().generate(' '.join(train['question_text']))
plt.imshow(wc)
insincere_rows = train[train['target'] == 1 ]

wc = WordCloud().generate(' '.join(insincere_rows['question_text']))
plt.imshow(wc)
# convert all characters to lower case

docs = train['question_text'].str.lower()
# apply regular expression to retain only alphabets and spaces

docs = docs.str.replace('[^a-z ]', '')
docs.head()
# remove commonly used words and apply stemming

import nltk
stopwords = nltk.corpus.stopwords.words('english')
custom_stopwords = ['will']
stopwords.extend(custom_stopwords)
len(stopwords)
# stemming

stemmer = nltk.stem.PorterStemmer()
def clean_sentences(text):
    words = text.split(' ')
    clean_words = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(clean_words)

docs_clean = docs.apply(clean_sentences)
docs_clean.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

train_model , validate_model = train_test_split(docs_clean , test_size = 0.3 , random_state = 100)
vectorizer = CountVectorizer(min_df=50)
vectorizer.fit(train_model)

train_dtm = vectorizer.transform(train_model)
validate_dtm = vectorizer.transform(validate_model)
train_dtm
train_model.shape
unique_terms = vectorizer.get_feature_names()
len(unique_terms)
unique_terms[:100]
df_dtm = pd.DataFrame(train_dtm[:10].toarray(),
                     columns = vectorizer.get_feature_names())
df_dtm
train_x = train_dtm
validate_x = validate_dtm

train_y = train.loc[train_model.index]['target']
validate_y = train.loc[validate_model.index]['target']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , f1_score , classification_report
model_dt = DecisionTreeClassifier(max_depth = 20)
model_dt.fit(train_x , train_y)
validate_pred = model_dt.predict(validate_x)
print(accuracy_score(validate_y,validate_pred))
print(f1_score(validate_y,validate_pred))
test = pd.read_csv('../input/test.csv')
docs_test = test['question_text'].str.lower()
docs_test = docs_test.str.replace('[^a-z ]','')
docs_test_clean = docs_test.apply(clean_sentences)

test_dtm = vectorizer.transform(docs_test_clean)
test_pred = model_dt.predict(test_dtm)
submission = pd.DataFrame({ 
    'qid' : test['qid'],
    'prediction' : test_pred})
submission.to_csv('submission.csv',index = False)
