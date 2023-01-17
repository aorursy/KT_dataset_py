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
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train.shape)

print(test.shape)
train.info()
train_data = train.drop(['keyword', 'location', 'id'], axis=1)
import re

def  clean_text(df, text_field):

    df[text_field] = df[text_field].str.lower()

    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    return df

data_clean = clean_text(train_data, "text")

data_clean.head()
stop = stopwords.words('english')

data_clean['text'] = data_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean.head()
X_train, X_test, y_train, y_test = train_test_split(data_clean['text'],data_clean['target'],random_state = 42)
pipeline = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf',  TfidfTransformer()),

    ('svm', LinearSVC()),

])

model = pipeline.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f1_score(y_test, y_predict))
parameters = {

    'svm__C': (0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5)

}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Best score: %0.3f" % grid_search.best_score_)
submission_test_clean = test.copy()

submission_test_clean = clean_text(submission_test_clean, 'text')

submission_test_clean['text'] = submission_test_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

submission_test_clean = submission_test_clean['text']

submission_test_clean.head()
pred = grid_search.predict(submission_test_clean)
id_col = test['id']

submission = pd.DataFrame({

                  "id": id_col, 

                  "target": pred})

submission.head()
submission.to_csv('submission_1.csv', index=False)