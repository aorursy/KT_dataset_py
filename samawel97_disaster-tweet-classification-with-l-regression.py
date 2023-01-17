import pandas as pd

import numpy as np

import re 
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data  =pd.read_csv('../input/nlp-getting-started/test.csv')

train_data.head(10)
train_data.dtypes
train_data['text'][0]
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

from sklearn.linear_model import LogisticRegression

pipeline_sgd = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf',  TfidfTransformer()),

    ('lr', LogisticRegression()),

])

model = pipeline_sgd.fit(X_train, y_train)
from sklearn.metrics import classification_report

y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))
#Confusion Matrix Visualisation

import matplotlib.pyplot as plt 

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(model, X_test, y_test) 

plt.show()
test_data.head()
submission_test_clean = test_data.copy()

submission_test_clean = clean_text(submission_test_clean, "text","text_clean")

submission_test_clean['text_clean'] = submission_test_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

submission_test_clean.head()
submission_test_pred = model.predict(submission_test_clean['text_clean'])
id_col = test_data['id']

submission_df_kaggle = pd.DataFrame({"id": id_col,"target": submission_test_pred})

submission_df_kaggle.head()
submission_df_kaggle.to_csv("submission.csv", index=False)