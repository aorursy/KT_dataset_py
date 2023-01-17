# Updating pip and installing keras-tuner package
!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install -U keras-tuner
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kerastuner import HyperModel, Objective
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import keras.backend as K

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
train_df = train_df.fillna('empty')
test_df = test_df.fillna('empty')
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
                                                  filters='!"$%&()*+.,-/:;=?@[\]^_`{|}~# ')

# Fit the tokenizer on the trainig data.
tokenizer.fit_on_texts(train_df.text)
# Convert the texts to tokens
train_df['text_tokenized'] = tokenizer.texts_to_sequences(train_df.text)
test_df['text_tokenized'] = tokenizer.texts_to_sequences(test_df.text)

# Pad the sequence as we will like to have sentences of equal length.
# We can also use bucket with sequence length.
np_matrix_train = tf.keras.preprocessing.sequence.pad_sequences(train_df['text_tokenized'])
np_matrix_train = np.append(np_matrix_train, np.expand_dims(train_df['target'], axis=-1), axis=1)
# convert the data to x and y, Later batch the dataset.
train_dataset = tf.data.Dataset.from_tensor_slices(np_matrix_train)
train_dataset_all = train_dataset.map(lambda x: (x[:-1], x[-1])).batch(batch_size)



# Do the similar to the test dataset.
np_matrix_test = tf.keras.preprocessing.sequence.pad_sequences(test_df['text_tokenized'])
test_dataset = tf.data.Dataset.from_tensor_slices(np_matrix_test).batch(1)
def is_test(x, y):
    return x % 5 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

validation_dataset = train_dataset_all.enumerate() \
                    .filter(is_test) \
                    .map(recover)

train_dataset = train_dataset_all.enumerate() \
                    .filter(is_train) \
                    .map(recover)
!rm -rf  ./real_or_not*
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
# Create the keras tuner model.
class MyHyperModel(HyperModel):
    
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, embedding_dim))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(tf.keras.layers.Conv1D(filters=hp.Choice('num_filters', values=[32, 64], default=64),activation='relu',
                                             kernel_size=3,
                                             bias_initializer='glorot_uniform'))
            model.add(tf.keras.layers.MaxPool1D())
        
        model.add(tf.keras.layers.GlobalMaxPool1D())
        
        for i in range(hp.Int('num_layers_rnn', 1, 3)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=hp.Choice('optimizer', values= ['Adam', 'Adadelta', 'Adamax']),
            loss='binary_crossentropy',
            metrics=[f1])
        return model


hypermodel = MyHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective=Objective('val_f1', direction="max"),
    max_trials=15,
    directory='./',
    project_name='real_or_not')

tuner.search(train_dataset,
             epochs=10, validation_data=validation_dataset)

tuner.results_summary()
models = tuner.get_best_models(num_models=1)

models[0].summary()
models[0].predict(test_dataset)
result_dataframe = pd.DataFrame(columns=['id', 'target'])
result_dataframe['id'] = test_df['id']
result_dataframe['target'] = np.where(np.array(models[0].predict(test_dataset)) >= 0.5, 1, 0 )
result_dataframe.to_csv('result.csv', index= False)