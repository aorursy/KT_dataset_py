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
my_file = '../input/arxiv/arxiv-metadata-oai-snapshot-2020-08-14.json'
# get access to metadata
def get_metadata():
    with open(my_file, 'r') as f:
        for line in f:
            yield line
            
metadata = get_metadata()
import json
categories = []
abstract = []
for ind, paper in enumerate(metadata):
    paper = json.loads(paper)
    categories.append(paper['categories'])
    abstract.append(paper['abstract'])
# will only use the first category as a lable to simplify the problem
def clean_catagoris(catagorie):
    return catagorie.replace(' ','.').split('.')[0]

categories2 = []
for i in categories:
    categories2.append(clean_catagoris(i))
categories = pd.Series(categories2)

len(abstract)
# will only use 15% of the total data for training, to speed up training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    abstract, categories, test_size=0.85, random_state=42)
# turn y_train to catagories
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_num = le.transform(y_train)
len(np.unique(y_train))
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow as tf

# a neural network with pretrained embedding layer, the embedding layer also is trainable

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50],trainable=True),
    keras.layers.Dropout(rate=.4),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(rate=.4),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(rate=.4),
    keras.layers.Dense(38, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)
history = model.fit(np.array(X_train),y_train_num, validation_split=.2, epochs=30,
                    callbacks=[early_stopping_cb])
le.inverse_transform(model.predict_classes(X_test[:15]))
y_pred = le.transform(y_test[:5000])
model.evaluate(np.array(X_test[:5000]),y_pred)
y_test[:15].values
#an example of a wrong prediction
X_test[3]
from sklearn.metrics import confusion_matrix
pred = model.predict_classes(X_test[:5000])

cf_matrix = confusion_matrix(y_pred,pred)
import seaborn as sns
sns.heatmap(cf_matrix/np.sum(cf_matrix))
