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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',dtype={'id': np.int16, 'target': np.int8})
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',dtype={'id': np.int16})
print('Training Set Shape = {}'.format(train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(train.memory_usage().sum() / 1024**2))
print('Training Set columns = {}'.format(train.columns))
print('Test Set columns = {}'.format(test.columns))
print('Test Set Shape = {}'.format(test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(test.memory_usage().sum() / 1024**2))
train.head(5)
test.head(5)
train.isnull().sum()
test.isnull().sum()

x=train.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')

def create_corpus(train):
    corpus=[]
    
    for x in train['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus
train_data = train.drop(['keyword', 'location', 'id'], axis=1)
train_data.head()
import re
def  clean_text(df, text_field,keyword,location):
    df[keyword] = df[keyword].str.lower()
    df[keyword] = df[keyword].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
   
    df[location] = df[location].str.lower()
    df[location] = df[location].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df

train = train.fillna('.')
test = test.fillna('.')
data_clean = clean_text(train, "text","keyword","location")
data_clean.head()

corpus_n=create_corpus(train_data)
print(len(corpus))
with open("corpus.txt",'w') as f:
    for item in corpus_n:
        f.write("%s\n"%item)
from nltk.corpus import stopwords
stop = stopwords.words('english')
data_clean['text'] = data_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_clean['keyword'] = data_clean['keyword'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_clean['location'] = data_clean['location'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_clean.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_clean['text'],data_clean['target'],random_state = 0)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', SGDClassifier()),
])
pipeline_nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', MultinomialNB()),])
pipeline_lr = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', LogisticRegression()),])
model_sgd = pipeline_sgd.fit(X_train, y_train)
model_lr = pipeline_lr.fit(X_train, y_train)
model_nb = pipeline_nb.fit(X_train, y_train)
from sklearn.metrics import classification_report
y_predict_sgd = model_sgd.predict(X_test)
print(classification_report(y_test, y_predict_sgd))
y_predict_lr = model_lr.predict(X_test)
print(classification_report(y_test, y_predict_lr))
y_predict_nb = model_nb.predict(X_test)
print(classification_report(y_test, y_predict_nb))
submission_test_clean = test.copy()
submission_test_clean = clean_text(submission_test_clean, "text","keyword","location")
submission_test_clean['text'] = submission_test_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
submission_test_clean = submission_test_clean['text']
submission_test_clean.head()
submission_test_pred=model_nb.predict(submission_test_clean)
id_col = test['id']
submission_df_1 = pd.DataFrame({
                  "id": id_col, 
                  "target": submission_test_pred})
submission_df_1.head()
submission_df_1.to_csv('submission_2.csv', index=False)
