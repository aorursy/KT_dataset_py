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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import zipfile
# Function to extracr 
def load_data_from_zip(zipfile_path, delimiter=","):
    with zipfile.ZipFile(zipfile_path, "r") as f:
        name = f.namelist()[0]
        with f.open(name) as zf:
            return pd.read_csv(zf, sep=delimiter)

train_df = load_data_from_zip("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", "\t")
test_df = load_data_from_zip("/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip", "\t")
print(train_df.shape)
print(test_df.shape)
train_df.head()
train_df.describe()
train_index_list, validation_index_list = train_test_split(\
                                    np.unique(train_df['SentenceId']),\
                                    test_size=0.2, random_state=42)

validation_df = train_df[train_df['SentenceId'].isin(validation_index_list)]
train_df = train_df[train_df['SentenceId'].isin(train_index_list)]
train_df.head()
print(train_df.shape)
print(validation_df.shape)
class Model(tf.keras.Model):
    
    def __init__(self, embed_url):
        super().__init__()
        self.embed = hub.load(embed_url)
        self.sequential = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(5)
        ])
            
    def call(self, nn_input):
        phrases = nn_input['Phrase'][:,0]
        embedding = self.embed(phrases)
        return self.sequential(embedding)       
# Using Universal Sentence Encoder 
model = Model('https://tfhub.dev/google/universal-sentence-encoder/4')  
model.compile(
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.optimizers.Adam(),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )
history = model.fit(x=dict(train_df), y=train_df['Sentiment'],
                    validation_data=(dict(validation_df), validation_df['Sentiment']),
                    epochs=25
          )
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
train_eval_result = model.evaluate(dict(train_df), train_df['Sentiment'])
validation_eval_result = model.evaluate(dict(validation_df), validation_df['Sentiment'])

print(f"Training set accuracy: {train_eval_result[1]}")
print(f"Validation set accuracy: {validation_eval_result[1]}")

validation_set_predictions = model.predict(dict(validation_df))
validation_set_predictions = tf.argmax(validation_set_predictions, axis=-1)
cm = tf.math.confusion_matrix(validation_df['Sentiment'], validation_set_predictions)
cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
import seaborn as sns
sns.heatmap(
    cm, annot=True,
    xticklabels=range(5),
    yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("True")
predictions = model.predict(dict(test_df))
predictions = tf.argmax(predictions, axis=-1)
test_df['Sentiment'] = predictions
test_df[['PhraseId', 'Sentiment']].to_csv("predictions.csv", index=False)
test_df.head()
