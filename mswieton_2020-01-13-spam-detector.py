# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
from keras.preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences
from keras import Sequential



from keras.layers import Embedding

from keras.layers import Conv1D

from keras.layers import SpatialDropout1D

from keras.layers import GlobalMaxPooling1D

from keras.layers import Dense



from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
PATH = '/kaggle/input/sms-spam-collection-dataset/'

!ls $PATH
sms_data = pd.read_csv(PATH+'spam.csv', encoding='latin_1')

sms_data
# checking for missing values



sms_data.isna().sum()
# dropping (almost) empty columns as not important



cols_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

sms_data.drop(columns=cols_to_drop, inplace=True)

sms_data
# renaming feature and target columns



feat_name = 'sms_text'

target_name = 'spam'

sms_data.columns = [target_name, feat_name]

sms_data
sms_data.isna().sum()
# checking (binary) target distribution



sms_data['spam'].value_counts()
# checking if all sms texts are unique



len(sms_data['sms_text'].unique()) == len(sms_data['sms_text'])
# fitting Tokenizer on the "sms_text" corpus



tokenizer = Tokenizer()

tokenizer.fit_on_texts(sms_data['sms_text'])
# showing learned vocabulary with indices



# tokenizer.word_index
num_tokens = len(tokenizer.word_index)

print('Encoded "sms_text" corpus with {} token indices'

      .format(num_tokens))
# showing a few sample encodings



tokenizer.texts_to_sequences(['we are your friends', 

                              'nothing last forever',                               

                              'how do you feel today'])
# showing example reverse encoding



tokenizer.sequences_to_texts([[1, 86, 3], [49, 22, 3]])
# encoding whole "sms_text" data

sequences = tokenizer.texts_to_sequences(sms_data['sms_text'])



# padding sequences for equal length

sequences = pad_sequences(sequences)



num_seq = sequences.shape[0]

len_seq = sequences.shape[1]



print('Encoded {} sequences and padded for equal length of {} tokens'

      .format(num_seq, len_seq))
# encoding (binary) target variable



sms_data['target'] = [1 if is_spam == 'spam' else 0 for is_spam in sms_data['spam']]

sms_data.head(20)
# checking if target and feature lengths match



sms_data['target'].shape[0] == sequences.shape[0]
# embedding layer parameters



input_dim = num_tokens + 1

output_dim = 64

input_length = len_seq



# convolutional (1D) layer parameters

filters = 256

kernel_size = 3
optimizer = 'adam'

loss = 'binary_crossentropy'

metrics = ['accuracy']
batch_size = 128

epochs = 10



# setting up the "EarlyStopping" callback

early_stop = EarlyStopping(monitor='val_loss', 

                           min_delta=0, 

                           patience=3, 

                           verbose=True, 

                           mode='auto', 

                           baseline=None, 

                           restore_best_weights=False)



callbacks = [early_stop]



validation_split = 0.20



# setting class weights for the loss function to adjust for class imbalance

# 'spam' is set to weight 8x more

class_weight = {0: 1, 1: 8}
model = Sequential()



model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

model.add(Conv1D(filters=filters, kernel_size=kernel_size))

model.add(SpatialDropout1D(rate=0.25))

model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# extracting target to a single array for simplicity



target = sms_data['target'].values

target
# training model with validation and early stopping



model.fit(x=sequences, y=target, 

          batch_size=batch_size, epochs=epochs, 

          verbose=True, callbacks=callbacks, 

          validation_split=validation_split, 

          class_weight=class_weight)
# showing history of 'accuracy'



plt.figure()

plt.plot(model.history.history['accuracy'], label='TRAIN ACC')

plt.plot(model.history.history['val_accuracy'], label='VAL ACC')

plt.legend()

plt.show()
# showing history of 'loss'



plt.figure()

plt.plot(model.history.history['loss'], label='TRAIN LOSS')

plt.plot(model.history.history['val_loss'], label='VAL LOSS')

plt.legend()

plt.show()
# making predictions for training sequences (in-sample check)



pred = model.predict_classes(sequences)

pred.shape
# showing confusion matrix



cm = confusion_matrix(y_true=target, y_pred=pred)

cm = pd.DataFrame(cm)

cm
# plotting the confusion matrix heatmap



plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True)
# setting the optimal number of epochs



epochs = early_stop.stopped_epoch + 1
model.fit(x=sequences, y=target, 

          batch_size=batch_size, epochs=epochs, 

          verbose=True, 

          class_weight=class_weight)
"""

helper function: check if a SMS text provided would be classified as spam or not

argument: <string> with SMS text to be checked

if no argument provided, read the user's input

"""



def check_if_spam(sms=None):



    # read user's input if no argument provided

    if sms is None:

        sms = input('Enter SMS text: ')

    

    # tokenize the SMS text and pad sequence to match training sequences length

    sms = [sms,]

    sequence = tokenizer.texts_to_sequences(sms)

    sequence = pad_sequences(sequence, maxlen=len_seq)

    

    # predict class and give feedback

    pred_class = model.predict_classes(sequence)

    is_spam = 'This is SPAM !!!' if pred_class == 1 else 'This is not spam.'

        

    return is_spam
my_sample = ['Final chance to win free tickets. Call now!', 

             'Suspicious activity detected. Follow this link to change password immediately.',

             'Get over here and call me tonite. Only 2 USD for minute.',

             'What are you waiting for! These are final days of our xmass promo deals.',

             'We have new offers for you. Visit our webpage and see.',

             'Binary FX options trading and 100 USD on your account. Hurry up.',

             'Huge discounts this weekend. Check this site to learn more.',

             'You can also earn easy money. Call us now.',

             'Congratulations! to claim your reward you must reply immediately',

             'For our database update we need a contact from you. Call us at.'

            ]



for text in my_sample:

    print('\nChecking:     ', text)

    print(check_if_spam(text))
# call the 'check_if_spam' function with no arguments to provide custom text

# uncomment to see in action



# check_if_spam()
# helper script to show random "spam message"



spams = sms_data[sms_data['spam'] == 'spam']

idx = np.random.randint(len(spams))

spam = spams.iloc[idx]['sms_text']

print(spam)