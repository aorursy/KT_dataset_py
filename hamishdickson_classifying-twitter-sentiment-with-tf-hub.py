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
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv('../input/Sentiment.csv')
df.head(5)
df.sentiment.value_counts()
df.sentiment_confidence.plot(kind='hist')
df = df[df['sentiment_confidence'] > 0.5]

df_pos = df[df['sentiment'] == 'Positive'].sample(frac=1)
df_neg = df[df['sentiment'] == 'Negative'].sample(frac=1)
df_neu = df[df['sentiment'] == 'Neutral'].sample(frac=1)

sample_size = min(len(df_pos), len(df_neg), len(df_neu))

df_ = pd.concat([df_pos.head(sample_size), df_neg.head(sample_size), df_neu.head(sample_size)])[['text', 'sentiment']]
del df
sample_size
import re

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = str(string)
    # remove the retweet part - maybe this should just be removed
    if string[:4] == 'RT @':
        tmp = string.find(':')
        string = string[tmp + 2:]
    string = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
df_['clean'] = df_['text'].apply(clean_str)
df_.head(3)
text_embeddings  = hub.text_embedding_column(
  "clean", 
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2"
)
X_train, X_test, y_train, y_test = train_test_split(df_['clean'], df_['sentiment'], test_size=0.3, random_state=42)
multi_class_head  = tf.contrib.estimator.multi_class_head(
    n_classes=3,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
    label_vocabulary=['Positive', 'Neutral', 'Negative']
)
estimator  = tf.contrib.estimator.DNNEstimator(
    head=multi_class_head,
    hidden_units=[256, 128, 64],
    feature_columns=[text_embeddings],
    optimizer=tf.train.AdamOptimizer()
)
features = {
  "clean": np.array(X_train)
}
labels = np.array(y_train)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, 
    labels, 
    shuffle=True, 
    batch_size=32, 
    num_epochs=20
)
estimator.train(input_fn=train_input_fn)
eval_input_fn  = tf.estimator.inputs.numpy_input_fn({"clean": np.array(X_test).astype(np.str)}, np.array(y_test), shuffle=False)

estimator.evaluate(input_fn=eval_input_fn)
