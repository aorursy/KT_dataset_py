import tensorflow as tf

import os 

import zipfile

import pandas as pd

import numpy as np
!pip install transformers==3.1.0
from transformers import DistilBertTokenizerFast

from transformers import TFDistilBertForSequenceClassification
import json



path="../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"



def parse_data(file):

    for l in open(file,'r'):

        yield json.loads(l)



data = list(parse_data(path))



training_size = 25000



sentences = []

labels = []

urls = []

for item in data:

    sentences.append(item['headline'])

    labels.append(item['is_sarcastic'])



training_sentences = sentences[0:training_size]

validation_sentences = sentences[training_size:]

training_labels = labels[0:training_size]

validation_labels = labels[training_size:]
len(training_sentences)
len(validation_sentences)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(training_sentences,

                            truncation=True,

                            padding=True)

val_encodings = tokenizer(validation_sentences,

                            truncation=True,

                            padding=True)
train_dataset = tf.data.Dataset.from_tensor_slices((

    dict(train_encodings),

    training_labels

))



val_dataset = tf.data.Dataset.from_tensor_slices((

    dict(val_encodings),

    validation_labels

))
# We classify two labels in this example. In case of multiclass classification, adjust num_labels value

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',

                                                              num_labels=2)
# Commented because it would take lot of time while committing

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

#model.fit(train_dataset.shuffle(42).batch(16),epochs=2,batch_size=16,validation_data=val_dataset.shuffle(42).batch(16))