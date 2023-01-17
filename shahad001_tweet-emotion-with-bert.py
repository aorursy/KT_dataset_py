!pip install transformers
import numpy as np # linear algebra

import pandas as pd # data processing

import tensorflow as tf



from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split



from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification



# for text cleaning

import string

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



# for building the model

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model





pd.set_option('display.max_colwidth', None)
tdata = pd.read_csv("../input/twitter-text-emotions/text_emotion.csv", usecols=[1,3])

print(tdata.describe())

print(tdata['sentiment'].value_counts())
stop_words = stopwords.words('english')

def clean_text(text):

    tokens = word_tokenize(text) # divide into tokens

    table = str.maketrans('', '', string.punctuation)

    words = [w.lower().translate(table) for w in tokens] # remove the punc'ns and convert to lower case

    words = [w for w in words if w.isalpha()]

    words = [w for w in words if not w in stop_words]

    

    clean_text = ' '.join(word for word in words)    

    return clean_text





def create_inputs(tweets, tokenizer):

    input_ids = []

    MAX_LEN = -1

    for tweet in tweets:

        clean_tweet = clean_text(tweet)

        clean_tweet ='[CLS] ' + clean_tweet + ' [SEP]'

        tokens = tokenizer.tokenize(clean_tweet)

        # convert to input ids

        inp_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids.append(inp_ids)

        # get the length

        size = len(inp_ids)

        MAX_LEN = max(MAX_LEN, size)

    

    # padding

    input_ids_padded = pad_sequences([inp_ids for inp_ids in input_ids], 

                              maxlen=MAX_LEN, truncating="post", padding="post")

    # creating the attention mask

    attention_masks = np.where(input_ids_padded != 0, 1, 0)

#     print(input_ids_padded.shape)

#     print(attention_masks.shape)

    return input_ids_padded, attention_masks, MAX_LEN
# create the BERT tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids, attention_masks, MAX_LEN = create_inputs(tdata['content'], tokenizer)

# the inputs contains the input_ids and the attention_masks
# prepare the outputs or labels

# tdata['sentiment'].value_counts().plot(kind='bar', stacked=False)

dummies = pd.get_dummies(tdata['sentiment']).values

# labels = tf.cast(dummies, tf.int32)

# print(labels[0].shape[0])
# train test dataset splitting

train_inp, test_inp, train_out, test_out, train_mask, test_mask = train_test_split(

    input_ids, dummies, attention_masks,test_size = 0.2, random_state = 50)
# create the model

inputs = Input((MAX_LEN, ), dtype=tf.int32, name="input_ids")

masks = Input((MAX_LEN, ), dtype=tf.int32, name="attention_masks")

bert_model = TFBertModel.from_pretrained('bert-base-uncased')

outs, _ = bert_model([inputs, masks])

out = GlobalAveragePooling1D()(outs)

out = Dense(100, activation='relu')(out)

outputs =  Dense(13, activation='softmax', name='output')(out)

model = Model([inputs, masks], outputs)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.summary()
train_inputs = [tf.cast(train_inp, tf.int32), tf.cast(train_mask, tf.int32)]

test_inputs = [tf.cast(test_inp, tf.int32), tf.cast(test_mask, tf.int32)]

train_target = tf.cast(train_out, tf.int32)

test_target = tf.cast(test_out, tf.int32)



val_inputs = train_inputs[:1000]

val_targets = train_target[:1000]

partial_inputs = train_inputs[1000:]

partial_targets = train_target[1000:]
history = model.fit(train_inputs, train_target, epochs=20, batch_size=32)
import matplotlib.pyplot as plt



loss = history.history['loss']

accu = history.history['accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')

plt.title('Training')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# validation

correct = 0

# print(test_inputs[0][0]) 

# len(test_inputs[0])

for i in range(len(test_inputs[0])):

    inp = [[test_inputs[0][i]], [test_inputs[1][i]]]

#     print(inp)

    result = model.predict(inp, batch_size=1)[0]

    

#     print('found:',np.argmax(result))

#     print('target:',np.argmax(test_target[i]))

    if np.argmax(result) == np.argmax(test_target[i]):

        correct += 1



print('Accurecy: ', (correct/len(test_inputs[0]))*100 ,'%' )