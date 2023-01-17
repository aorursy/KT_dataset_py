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

import re
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data  =pd.read_csv('../input/nlp-getting-started/test.csv')

train_data.head(10)

train_data.dtypes
train_data['text'][11]
import re

def  clean_text(df, text_field, new_text_field_name):

    df[new_text_field_name] = df[text_field].str.lower()

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    # remove numbers

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    

    return df

data_clean = clean_text(train_data, 'text', 'text_clean')

data_clean.head()
import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean.head()
import nltk 

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))

data_clean.head()
from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))

data_clean.head()
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))

data_clean.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_clean['text_clean'],data_clean['target'],random_state = 0)

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

submission_test_clean = clean_text(submission_test_clean, "text","text_clean")

submission_test_clean['text_clean'] = submission_test_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

submission_test_clean = submission_test_clean['text_clean']

submission_test_clean.head()
submission_test_pred = model.predict(submission_test_clean)
id_col = test_data['id']

submission_df_1 = pd.DataFrame({

                  "id": id_col, 

                  "target": submission_test_pred})

submission_df_1.head()
submission_df_1.to_csv('submission_1.csv', index=False)