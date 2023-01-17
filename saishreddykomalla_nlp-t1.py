from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = pd.read_csv('../input/nlp-getting-started/train.csv')
train_path = train_path.sample(frac=1)
train_path.head()

training_sentences, training_labels = [], []
def add_data( l, d):
    for i in train_path[d]:
        l.append(i)
add_data( training_sentences, 'text')
add_data( training_labels,  'target')
tokenizer = Tokenizer(oov_token = "<404>")
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training = tokenizer.texts_to_sequences(training_sentences)
train_padded = pad_sequences(training, padding = 'pre')
train_sentences_ = np.array(train_padded)

training_labels_ = np.array(training_labels)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index)+1, 16, input_length = 33),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation = 'relu'),
#     tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.summary()
history = model.fit(train_sentences_, training_labels_, epochs = 15, validation_split = 0.1)
import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
fig, ax = plt.subplots(2,1)
ax[0].plot(acc, 'b', label = "Training_accuracy")
ax[0].plot(val_acc, 'r', label = "dev_accuracy")
ax[0].legend(loc = 'best', shadow = True)
    
ax[1].plot(loss, 'b', label = "trianing_loss")
ax[1].plot(val_loss, 'r', label = "dev_loss")
ax[1].legend(loc = 'best', shadow = True)
# plot_graphs(acc, val_acc, loss, val_loss)
test_path = pd.read_csv('../input/nlp-getting-started/test.csv')
test_set = [x for x in test_path['text']]
test_sequences = tokenizer.texts_to_sequences(test_set)
test_padded = pad_sequences(test_sequences,maxlen = 33, truncating = 'post', padding = 'pre')
pred = model.predict(test_padded)
test_predictions = []
for i in pred.round().astype(int):
    test_predictions.append(i[0])
ids = [str(x) for x in test_path['id']]
sol = pd.DataFrame({'id':ids, 'target': test_predictions})
sol.to_csv('prediction3.csv', index = False)
#########################################################################
#########################################################################
single_test=[]
single_test.append(input())
single_test_seqs = tokenizer.texts_to_sequences(single_test)
single_test_padded = pad_sequences(single_test_seqs, maxlen = 33, truncating = 'post', padding = 'pre')
print("Disaster" if model.predict(single_test_padded).round() else "fake")