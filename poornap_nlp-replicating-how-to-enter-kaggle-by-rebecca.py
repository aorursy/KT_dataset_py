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

train_data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
len(train_data[train_data['target']==0]),len(train_data[train_data['target']==1])
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_data.head()
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample_submission.head()
train_data = train_data.drop(['keyword', 'location', 'id'], axis=1)

train_data.head()
import re

def  clean_text(df, text_field):

    df[text_field] = df[text_field].str.lower()

    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    return df

data_clean = clean_text(train_data, "text")

data_clean.head()
import nltk.corpus

#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data_clean['text'] = data_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_clean['text'],data_clean['target'],random_state = 0)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier

pipeline_sgd = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf',  TfidfTransformer()),

    ('nb', SGDClassifier()),

])

model = pipeline_sgd.fit(X_train, y_train)
from sklearn.metrics import classification_report

y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))

submission_test_clean = test_data.copy()

submission_test_clean = clean_text(submission_test_clean, "text")

submission_test_clean['text'] = submission_test_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

submission_test_clean = submission_test_clean['text']

submission_test_clean.head()
submission_test_pred = model.predict(submission_test_clean)
id_col = test_data['id']

submission_df_1 = pd.DataFrame({

                  "id": id_col, 

                  "target": submission_test_pred})

submission_df_1.head()
submission_df_1.to_csv('submission_1.csv', index=False)