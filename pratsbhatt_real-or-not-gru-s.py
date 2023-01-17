# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Initializing the embedding dimention
batch_size = 64
embedding_dim = 512

# Read the input data.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Fill all the na data with empty character.
train_df = train_df.fillna('')
test_df = test_df.fillna('')
# Join with the text, keyword as well as the location
train_df['text'] = train_df['text'] + ' ' + train_df['keyword'].astype(str) + ' ' + train_df['location'].astype(str)
test_df['text'] = test_df['text'] + ' ' + test_df['keyword'].astype(str) + ' ' + test_df['location'].astype(str)

# Strip off the whitespace from the front and back of the sentence
train_df['text'] = train_df['text'].str.strip()
test_df['text'] = test_df['text'].str.strip()

# Replace all the links, with just link as word.
# It matters more to know if there is a link or not in place of which link it is actually.
# But this remains to be seen later.
train_df['text'] = train_df['text'].str.replace(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'link')
test_df['text'] = test_df['text'].str.replace(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'link')
# To see the output of the pre-processing steps.
train_df.to_csv('train.csv', index=False)

# Drop the kexword and location column as it is not with the text column
train_df.drop(columns=['keyword', 'location'])

# Use the tokenizer and remove all the special characters. Add oov_character as irrelevant
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="'<irrelevant>'",
                                                  filters='!"$%&()*+.,-/:;=?@[\]^_`{|}~ ')

# Fit the tokenizer on the trainig data.
tokenizer.fit_on_texts(train_df.text)
# Convert the texts to tokens
train_df['text_tokenized'] = tokenizer.texts_to_sequences(train_df.text)
test_df['text_tokenized'] = tokenizer.texts_to_sequences(test_df.text)

# Pad the sequence as we will like to have sentences of equal length.
# We can also use bucket with sequence length.
np_matrix_train = tf.keras.preprocessing.sequence.pad_sequences(train_df['text_tokenized'])
np_matrix_train = np.append(np_matrix_train, np.expand_dims(train_df['target'], axis=-1), axis=1)
# convert the data to x and y, later shuffle the data each iteration. Later batch the dataset.
train_dataset = tf.data.Dataset.from_tensor_slices(np_matrix_train)
train_dataset = train_dataset.map(lambda x: (x[:-1], x[-1])).shuffle(7000, reshuffle_each_iteration=True).batch(batch_size)

# Do the similar to the test dataset.
np_matrix_test = tf.keras.preprocessing.sequence.pad_sequences(test_df['text_tokenized'])
test_dataset = tf.data.Dataset.from_tensor_slices(np_matrix_test).batch(1)
# Creating keras model using two GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embedding_dim),
    tf.keras.layers.GRU(40, return_sequences=True),
    tf.keras.layers.GRU(40),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    
model.summary()

# Using adam as optimizer
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['binary_accuracy'])

# early stopping the training if the loss is not decreasing with patience value as 5.
callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

# scheduling the learning rate.
def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

learning_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# training the model
history = model.fit(train_dataset, epochs=200, callbacks=[learning_callback])

# predicting the result on the training dataset.
result_dataframe = pd.DataFrame(columns=['id', 'target'])
result_dataframe['id'] = test_df['id']
result_dataframe['target'] = np.where(np.array(model.predict(test_dataset)) > 0.5, 1, 0 )
result_dataframe.to_csv('result.csv', index= False)
