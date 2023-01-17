import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from keras.models import model_from_json
import pickle
#Open csv & add header
data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', engine = 'python', header = None)
data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
set(data.target)
data.target = (data.target).replace(4,1)
set(data.target)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.text, data.target, test_size=0.10, random_state=48)
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
print('prepearing train and test...')
# Train
for s in X_train:
  training_sentences.append(s)
for l in y_train:  
  training_labels.append(l)
#Test
for s in X_test:
  testing_sentences.append(s)
for l in y_test:  
  testing_labels.append(l)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
vocab_size = 30000
embedding_dim = 200
max_length = 120
trunc_type = 'post'
oov_tok =  '<OOV>'
tokenizer = Tokenizer(num_words = vocab_size, oov_token= oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

with tpu_strategy.scope():
    model = tf.keras.Sequential([
                                 tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
                                 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dropout((0.2)),
                                 tf.keras.layers.Dense(64, activation='relu'),
                                 tf.keras.layers.Dense(1, activation= 'sigmoid')
    ])

    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'),
                  metrics=['accuracy'])
num_epochs = 5
history = model.fit(
    padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    batch_size = 16 * tpu_strategy.num_replicas_in_sync
)
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
preds = model.predict(testing_padded)
print(classification_report(testing_labels, preds.round()))
num_epochs = 2
history = model.fit(
    padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    batch_size = 16 * tpu_strategy.num_replicas_in_sync
)
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
preds = model.predict(testing_padded)
print(classification_report(testing_labels, preds.round()))
filepath = 'model.json'
print('saving model...')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

outfile = open('tokenizer.pkl','wb')
pickle.dump(tokenizer,outfile)
outfile.close()
print('Tokenizer saved!')
print('model saved!')
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(testing_labels, preds, pos_label=None)
thresholds
from sklearn.metrics import roc_curve
from numpy import sqrt
from numpy import argmax
fpr, tpr, thresholds = roc_curve(testing_labels, preds)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
from matplotlib import pyplot
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
NEUTRAL='neutral'
NEGATIVE='negative'
POSITIVE='positive'
SENTIMENT_THRESHOLD = thresholds[ix]
trunc_type = 'post'
import time
def decode_sentiment(score):
    return NEGATIVE if score <= SENTIMENT_THRESHOLD else POSITIVE
def predict(text):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=max_length,
        truncating=trunc_type)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at} 
res = predict(list("I suck at Mortal Kombat"))

print(res)
res = predict(list("Your code is without doubt the worst code I've ever seen"))

print(res)
res = predict(list("I love this song!"))
print(res)
res = predict(list("I love cats"))
print(res)
res = predict(list("I feel lucky today"))
print(res)
res = predict(list("I hate storms"))
print(res)
res = predict(list("this is a terrible moment"))
print(res)
res = predict(list("I've failed my German exam"))
print(res)
res = predict(list("The code looks clean"))
print(res)
res = predict(list("you won't get that degree"))
print(res)