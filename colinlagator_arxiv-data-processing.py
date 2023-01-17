import numpy as np

import pandas as pd

import json

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

import plotly.express as px

import re



import torch

import tensorflow as tf

from sklearn.model_selection import train_test_split
def get_metadata():

    with open('../input/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as f:

        for line in f:

            yield line
metadata = get_metadata()
titles_tags_dict = {"title":[], "tags":[]}

for paper in metadata:

    parsed = json.loads(paper)

    titles_tags_dict["title"].append(parsed['title'])

    titles_tags_dict["tags"].append(parsed['categories'])
titles_tags_df = pd.DataFrame.from_records(titles_tags_dict)
titles_tags_df.head(5)
categories = titles_tags_df['tags'].apply(lambda x: x.split(' ')).explode().unique()
label_to_int_dict = {}

for i, key in enumerate(categories):

    label_to_int_dict[key] = i
int_to_label_dict = {}

for key, val in label_to_int_dict.items():

    int_to_label_dict[val] = key
def generate_label_array(label):

    result = np.zeros(len(label_to_int_dict))

    labels = label.split(' ')

    for l in labels:

        result[label_to_int_dict[l]] = 1

    return np.expand_dims(result, 0)
tag_labels = [generate_label_array(tag) for tag in titles_tags_df["tags"]]
tag_labels = np.concatenate(tag_labels, axis = 0)
tag_labels[1]
titles = titles_tags_df['title'].apply(lambda x : x.lower())

titles = titles.apply(lambda x: re.sub('[^A-Za-z\s]+', ' ', x))

titles = titles.apply(lambda x: re.sub('\n', ' ', x))

titles = titles.apply(lambda x: re.sub(r'/s+', ' ', x))

titles = titles.apply(lambda x: re.sub(r'^/s', '', x))

titles = titles.apply(lambda x: re.sub(r'/s$', '', x))
title_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
title_tokenizer.fit_on_texts(titles)
title_tokens = title_tokenizer.texts_to_sequences(titles)
title_tokens = tf.keras.preprocessing.sequence.pad_sequences(title_tokens, maxlen=20, padding='post')
title_tokens[1]
tag_labels.shape[0] == title_tokens.shape[0]
x_train, x_test, y_train, y_test = train_test_split(title_tokens, tag_labels, test_size = 0.2)