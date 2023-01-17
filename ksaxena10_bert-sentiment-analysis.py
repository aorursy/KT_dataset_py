# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install tensorflow-gpu==2.0.0

!pip install ktrain
import ktrain 

from ktrain import text
df = pd.read_csv('/kaggle/input/nlp-dataset-collected-from-youtube-comments/iran.csv')
df.head()
df = df.dropna()
df.shape
df.label.value_counts().plot.bar()
x_train = df["Comments"].head(500)

y_train=  df["label"].head(500).values.tolist()

x_test = df["Comments"].tail(27)

y_test = df["label"].head(27).values.tolist()
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,

                                                                       x_test=x_test, y_test=y_test,

                                                                       class_names=['1', '0'],

                                                                       preprocess_mode='bert',

                                                                       ngram_range=1, 

                                                                       maxlen=350)
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
model.summary()
hist = learner.fit_onecycle(2e-5, 2) 