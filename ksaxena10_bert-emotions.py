!pip install tensorflow-gpu==2.0.0

!pip install ktrain

import numpy as np # linear algebra

import pandas as pd

import ktrain 

from ktrain import text # data processing, CSV file I/O (e.g. pd.read_csv)





#Loading the dataset

dataset = pd.read_csv("./../input/emotion-classification/emotion.data", nrows=5000)
dataset.emotions.value_counts().plot.bar()
# Construction of label2id and id2label dicts

labels = dataset["emotions"].values.tolist()

label2id = {l: i for i, l in enumerate(set(labels))}

id2label = {v: k for k, v in label2id.items()}

id2label
x_train = dataset["text"].head(4500)

y_train= [label2id[label] for label in dataset["emotions"].head(4500).values.tolist()]

x_test = dataset["text"].tail(500)

y_test = [label2id[label] for label in dataset["emotions"].tail(500).values.tolist()]
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,

                                                                       x_test=x_test, y_test=y_test,

                                                                       class_names=dataset['emotions'].unique().tolist(),

                                                                       preprocess_mode='bert',

                                                                       ngram_range=1, 

                                                                       maxlen=350, 

                                                                       max_features=35000)
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
model.summary()
hist = learner.fit_onecycle(2e-5, 1) 
learner.validate(val_data=(x_test, y_test), class_names=dataset['emotions'].unique().tolist())
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()