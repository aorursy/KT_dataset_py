import numpy as np 

import pandas as pd

import re

import matplotlib.pyplot as plt 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


print('{} rows and {} cols in training dataset.'.format(train_data.shape[0], train_data.shape[1]))

print('{} rows and {} cols in testing dataset.'.format(test_data.shape[0], test_data.shape[1]))
# pd.set_option('display.max_colwidth', -1)

train_data.head()
test_data.head()
sample_submission.head()
count_table = train_data.target.value_counts()

display(count_table)
plt.bar('False', count_table[0], label='False', width=0.5)

plt.bar('True', count_table[1], label='True', width=0.5)

plt.legend()

plt.title('Class Distribution')

plt.show()
train_data = train_data.drop(['keyword', 'location', 'id'], axis=1)

train_data.head()
def clean_text(df, text_field):

    df[text_field] = df[text_field].str.lower()

    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))

    return df



data_clean = clean_text(train_data, "text")



data_clean.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(train_data['text'])



x = vectorizer.transform(train_data['text']).todense()

y = train_data['target'].values



x.shape, y.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)
model = LogisticRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
test_x = vectorizer.transform(test_data['text']).todense()

test_x.shape
sample_submission['target']  = model.predict(test_x)

sample_submission.head()
sample_submission.to_csv('submission.csv', index = False)