import numpy as np

import pandas as pd

import csv

from bs4 import BeautifulSoup



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RES_DIR = "../input/"
# Load train data

def load_train_data(skip_content=False):

    categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']

    train_data = []

    for cat in categories:

        if skip_content:

            data = pd.read_csv("{}{}.csv".format(RES_DIR, cat), usecols=['id', 'title', 'tags'])

        else:

            data = pd.read_csv("{}{}.csv".format(RES_DIR, cat))

        data['category'] = cat

        train_data.append(data)

    

    return pd.concat(train_data)
def load_test_data():

    test_data = pd.read_csv(RES_DIR + 'test.csv')

    return test_data
train_data = load_train_data()

train_data.head()
test_data = load_test_data()

test_data.head()
def merge(row):

    title = row['title']

    content = row['content']

    clean_content = BeautifulSoup(content, "html.parser")

    clean_content = clean_content.get_text()

    row['text'] = title + " " + clean_content

    return row
#nlp_train_data = train_data.apply(merge, axis=1)[['id', 'text', 'tags']]

#nlp_train_data.head()
nlp_test_data = test_data.apply(merge, axis=1)[['id', 'text']]

nlp_test_data.head()
tfidf = TfidfVectorizer(analyzer = "word", max_features = 5000, 

                        stop_words="english", ngram_range=(1,2))

features = tfidf.fit_transform(nlp_test_data['text']).toarray()
## Select top features for each test sample
tfidf_tags = []

top_n = -5

feature_array = np.array(tfidf.get_feature_names())

tfidf_sorting = np.argsort(features)

for i, e in enumerate(tfidf_sorting):

    tmp_tags = []

    indexes = e[top_n:]

    for idx in indexes:

        cur_tag = feature_array[idx]

        if features[i][idx] > 0.1 and len(cur_tag)>3 and '_' not in cur_tag:

            tmp_tags.append(cur_tag.replace(' ', '-'))

    tfidf_tags.append(" ".join(tmp_tags))
df = pd.DataFrame({'id':test_data['id'], 'tags':tfidf_tags})

df.head()
df.to_csv('submission.csv', index=False, quoting=csv.QUOTE_ALL)