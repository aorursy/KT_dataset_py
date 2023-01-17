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
PATH = '/kaggle/input/sms-spam-collection-dataset/'

!ls $PATH
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

tf.__version__
from tensorflow.keras.models import Model



from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping



import tensorflow_hub as hub
# importing the official tokenization script created by the Google team

!wget https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py



import tokenization
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



# encoding (binary) target variable

sms_data['target'] = [1 if is_spam == 'spam' else 0 for is_spam in sms_data['spam']]



sms_data
sms_data.isna().sum()
# checking (binary) target distribution



sms_data['spam'].value_counts()
# checking if all sms texts are unique



len(sms_data['sms_text'].unique()) == len(sms_data['sms_text'])
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
# setting model optimization parameters:

optimizer = Adam(lr=2e-6)

loss = 'binary_crossentropy'

metrics = ['accuracy']
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)

bert_layer
# setting the tokenizer parameters

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()



tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# encoding "train" and "test" datasets with the 'bert_encode' helper function



train_input = bert_encode(sms_data['sms_text'].values, tokenizer, max_len=189)



num_seq = train_input[0].shape[0]

len_seq = train_input[0].shape[1]



print('Encoded {} sequences and padded for equal length of {} tokens'

      .format(num_seq, len_seq))
train_labels = sms_data['target'].values
model = build_model(bert_layer, max_len=189)

model.summary()
batch_size = 16

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
# training model with validation and early stopping



model.fit(x=train_input, y=train_labels, 

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



predictions = model.predict(train_input)

predictions.shape
pred_classes = (predictions > 0.5).astype(int)

pred_classes.shape
# showing confusion matrix



cm = confusion_matrix(y_true=train_labels, y_pred=pred_classes)

cm = pd.DataFrame(cm)

cm
# plotting the confusion matrix heatmap



plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True)
# setting the optimal number of epochs



epochs = early_stop.stopped_epoch + 1
model.fit(x=train_input, y=train_labels, 

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

    sequence = bert_encode(sms, tokenizer, max_len=189)

        

    # predict class and give feedback

    prediction = model.predict(sequence)

    pred_class = (prediction > 0.5).astype(int)

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