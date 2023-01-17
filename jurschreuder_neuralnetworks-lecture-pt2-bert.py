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
# install tesorflow bert package

!pip install bert-for-tf2



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers

import bert



#Loding pretrained bert layer

BertTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",

                            trainable=False)





# Loading tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = BertTokenizer(vocab_file, do_lower_case)



print("done!")
text = 'Encoding will be clear with this example'

# tokenize

tokens_list = tokenizer.tokenize(text)

print('Text after tokenization')

print(tokens_list)



# initilize dimension

max_len =12

text = tokens_list[:max_len-2]

input_sequence = ["[CLS]"] + text + ["[SEP]"]

print("After adding  flasges -[CLS] and [SEP]: ")

print(input_sequence)





tokens = tokenizer.convert_tokens_to_ids(input_sequence )

print("tokens to id ")

print(tokens)



pad_len = max_len -len(input_sequence)

tokens += [0] * pad_len

print("tokens: ")

print(tokens)



print(pad_len)

pad_masks = [1] * len(input_sequence) + [0] * pad_len

print("Pad Masking: ")

print(pad_masks)



segment_ids = [0] * max_len

print("Segment Ids: ")

print(segment_ids)
import numpy as np

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



MAX_LEN = 12



# encode train set 

train_input = bert_encode([

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is amazing',

    'this is a depressed text',

    'sure is nice this weather',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'so good to be outside!',

    'this is sad',

    'this is good',

    'I am crying a lot',

    'good to see you!',

    'this is sad',

    'this is good',

    'ah its raining again',

    'this is a happy text',

    'feeling a bit down',

    'wow so amazing!',

    'this is a depressed text',

    'this looks really amazing',

    'nobody likes me',

    'I feel great today',

    'lying in bed the whole day',

    'this is a happy text',

    'this is sad',

    'this is so nice',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good',

    'this is a depressed text',

    'this is a happy text',

    'this is sad',

    'this is good'

], tokenizer, max_len=MAX_LEN)

train_labels = np.array([

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],

    [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],



])



print("number of test samples:", len(train_input[0]), "labels:", len(train_labels))

# first define input for token, mask and segment id  

from tensorflow.keras.layers import  Input

input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")

input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")

segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")



#  output  

from tensorflow.keras.layers import Dense

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])  

clf_output = sequence_output[:, 0, :]

out = Dense(2, activation='softmax')(clf_output)



# intilize model

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# train

train_history = model.fit(

    train_input, train_labels,

    validation_split=0.1,

    epochs=2,

    batch_size=1

)



model.save('model.h5')

print("done and saved!")
test_input = bert_encode(['I feel sad', 'I am happy'], tokenizer, max_len= MAX_LEN )

test_pred = model.predict(test_input)

preds = np.around(test_pred, 3)

preds