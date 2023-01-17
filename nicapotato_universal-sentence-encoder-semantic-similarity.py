# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================
%%capture

# Install the latest Tensorflow version.

!pip3 install --upgrade tensorflow-gpu

# Install TF-Hub.

!pip3 install tensorflow-hub

!pip3 install seaborn
#@title Load the Universal Sentence Encoder's TF Hub module

from absl import logging



import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import re

import seaborn as sns



module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]

model = hub.load(module_url)

print ("module %s loaded" % module_url)

def embed(input):

  return model(input)
#@title Compute a representation for each message, showing various lengths supported.

word = "Elephant"

sentence = "I am a sentence for which I would like to get its embedding."

paragraph = (

    "Universal Sentence Encoder embeddings also support short paragraphs. "

    "There is no hard limit on how long the paragraph is. Roughly, the longer "

    "the more 'diluted' the embedding will be.")

messages = [word, sentence, paragraph]



# Reduce logging output.

logging.set_verbosity(logging.ERROR)



message_embeddings = embed(messages)



for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):

  print("Message: {}".format(messages[i]))

  print("Embedding size: {}".format(len(message_embedding)))

  message_embedding_snippet = ", ".join(

      (str(x) for x in message_embedding[:3]))

  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
def plot_similarity(labels, features, rotation):

  corr = np.inner(features, features)

  sns.set(font_scale=1.2)

  g = sns.heatmap(

      corr,

      xticklabels=labels,

      yticklabels=labels,

      vmin=0,

      vmax=1,

      cmap="YlOrRd")

  g.set_xticklabels(labels, rotation=rotation)

  g.set_title("Semantic Textual Similarity")



def run_and_plot(messages_):

  message_embeddings_ = embed(messages_)

  plot_similarity(messages_, message_embeddings_, 90)
messages = [

    # Smartphones

    "I like my phone",

    "My phone is not good.",

    "Your cellphone looks great.",



    # Weather

    "Will it snow tomorrow?",

    "Recently a lot of hurricanes have hit the US",

    "Global warming is real",



    # Food and health

    "An apple a day, keeps the doctors away",

    "Eating strawberries is healthy",

    "Is paleo better than keto?",



    # Asking about age

    "How old are you?",

    "what is your age?",

]



run_and_plot(messages)

               
import pandas

import scipy

import math

import csv



sts_dataset = tf.keras.utils.get_file(

    fname="Stsbenchmark.tar.gz",

    origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",

    extract=True)

sts_dev = pandas.read_table(

    os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"),

    error_bad_lines=False,

    skip_blank_lines=True,

    usecols=[4, 5, 6],

    names=["sim", "sent_1", "sent_2"])

sts_test = pandas.read_table(

    os.path.join(

        os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"),

    error_bad_lines=False,

    quoting=csv.QUOTE_NONE,

    skip_blank_lines=True,

    usecols=[4, 5, 6],

    names=["sim", "sent_1", "sent_2"])

# cleanup some NaN values in sts_dev

sts_dev = sts_dev[[isinstance(s, str) for s in sts_dev['sent_2']]]
sts_data = sts_dev #@param ["sts_dev", "sts_test"] {type:"raw"}



def run_sts_benchmark(batch):

  sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_1'].tolist())), axis=1)

  sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_2'].tolist())), axis=1)

  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)

  clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)

  scores = 1.0 - tf.acos(clip_cosine_similarities)

  """Returns the similarity scores"""

  return scores



dev_scores = sts_data['sim'].tolist()

scores = []

for batch in np.array_split(sts_data, 10):

  scores.extend(run_sts_benchmark(batch))



pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)

print('Pearson correlation coefficient = {0}\np-value = {1}'.format(

    pearson_correlation[0], pearson_correlation[1]))