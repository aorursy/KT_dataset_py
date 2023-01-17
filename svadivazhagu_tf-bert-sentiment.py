#Import the necessary libraries



import os

import gc

import numpy as np

import pandas as pd

#tensorflow imports

import tensorflow as tf

import tensorflow.keras.backend as K

import tokenizers

#Get the BERT text tokenizer and associated model for tensorflow

from transformers import BertTokenizer, BertConfig, TFBertModel

#tqdm to show progress throughout iterations

from tqdm import tqdm

#regex library

import re



#Allow support for loading bars in Pandas - this is just helpful

tqdm.pandas()
#Competition data inside the Kaggle kernel is located inside tweet-sentiment-extraction

#Global variables in python are capitalized

DATA = "/kaggle/input/tweet-sentiment-extraction/"

#load training set

train = pd.read_csv(DATA+'train.csv')

#load testing set

test = pd.read_csv(DATA+'test.csv')

#load sample submission to get the format for the final data submission

submission = pd.read_csv(DATA+'sample_submission.csv')
class config:

    #Max length of a tweet is 128

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 32

    TEST_BATCH_SIZE = 32

    EPOCHS = 10

    #Add the location of the model's UNCAPTIALIZED configuration

    BERT_CONFIG = '/kaggle/input/bertconfig/bert-base-uncased-config.json'

    BERT_PATH = "/kaggle/input/bert-base-uncased-huggingface-transformer/"

    TOKENIZER = tokenizers.BertWordPieceTokenizer("/kaggle/input/bert-base-uncased-huggingface-transformer//bert-base-uncased-vocab.txt", 

        lowercase=True)

    SAVEMODEL_PATH = '/kaggle/input/tftweetfinetuned/finetuned_bert.h5'

    THRESHOLD = 0.4
def process_data(tweet, selected_text, tokenizer):

    len_st = len(selected_text)

    idx0 = None

    idx1 = None

    

    #Go through the tweet and its selected text and see where the common words exist

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):

        if tweet[ind: ind+len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st

            break

    

    char_targets = [0] * len(tweet)

    if idx0 != None and idx1 != None:

        for ct in range(idx0, idx1):

            char_targets[ct] = 1

    #Tokenize the string

    tok_tweet = tokenizer.encode(tweet)

    input_ids_orig = tok_tweet.ids

    tweet_offsets = tok_tweet.offsets



    target_idx = []

    for j, (offset1, offset2) in enumerate(tweet_offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)

    #Find target words and return them

    targets = [0] * len(input_ids_orig)

    for idx in target_idx:

        targets[idx] = 1

    return targets
def cleanText(tweet):

    #list of emoji patterns appearing in the tweets to be removed

    emoji_pattern = re.compile("["

                               u"\U0001F600-\U0001F64F"  # emoticons

                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                               u"\U0001F680-\U0001F6FF"  # transport & map symbols

                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                               "]+", flags=re.UNICODE)

    text = str(tweet)

    # Remove emojis

    text = emoji_pattern.sub(r'', text)

    # Remove twitter handles (@___)

    text = re.sub(r'@\w+', '', text)

    # Remove links after research that t.co uses http still

    text = re.sub(r'http.?://[^/s]+[/s]?', '', text)

    return text.strip().lower()
train['text'] = train['text'].apply(lambda x: cleanText(x))
train['targets'] = train.progress_apply(lambda row: process_data(   str(row['text']), 

                                                                    str(row['selected_text']),

                                                                    config.TOKENIZER),

                                                                    axis=1)
train['targets'] = train['targets'].apply(lambda x :x + [0] * (config.MAX_LEN-len(x)))
def convert_to_transformer_inputs(text, tokenizer, max_sequence_length):

    inputs = tokenizer.encode(text)

    input_ids =  inputs.ids

    input_masks = inputs.attention_mask

    input_segments = inputs.type_ids

    padding_length = max_sequence_length - len(input_ids)

    padding_id = 0

    input_ids = input_ids + ([padding_id] * padding_length)

    input_masks = input_masks + ([0] * padding_length)

    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]
def compute_input_arrays(df, tokenizer, max_sequence_length):

    input_ids, input_masks, input_segments = [], [], []

    for _, instance in tqdm(df.iterrows()):

        ids, masks, segments= convert_to_transformer_inputs(str(instance.text),tokenizer, max_sequence_length)

        input_ids.append(ids)

        input_masks.append(masks)

        input_segments.append(segments)

        

    return [np.asarray(input_ids, dtype=np.int32), 

            np.asarray(input_masks, dtype=np.int32), 

            np.asarray(input_segments, dtype=np.int32)]
def compute_output_arrays(df, columns):

    return np.asarray(df[columns].values.tolist())
outputs = compute_output_arrays(train,'targets')

inputs = compute_input_arrays(train, config.TOKENIZER, config.MAX_LEN)

test_inputs = compute_input_arrays(test, config.TOKENIZER, config.MAX_LEN)
def create_model():

    ids = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)

    mask = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)

    attn = tf.keras.layers.Input((config.MAX_LEN,), dtype=tf.int32)

    bert_conf = BertConfig() 

    bert_model = TFBertModel.from_pretrained(config.BERT_PATH+'/bert-base-uncased-tf_model.h5', config=bert_conf)

    

    output = bert_model(ids, attention_mask=mask, token_type_ids=attn)

    

    out = tf.keras.layers.Dropout(0.1)(output[0]) 

    out = tf.keras.layers.Conv1D(1,1)(out)

    out = tf.keras.layers.Flatten()(out)

    out = tf.keras.layers.Activation('sigmoid')(out)

    model = tf.keras.models.Model(inputs=[ids, mask, attn], outputs=out)

    return model
K.clear_session()

model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
if not os.path.exists(config.SAVEMODEL_PATH):

    model.fit(inputs,outputs, epochs=config.EPOCHS, batch_size=config.TRAIN_BATCH_SIZE)

    model.save_weights(f'finetuned_bert.h5')

else:

    model.load_weights(config.SAVEMODEL_PATH)
predictions = model.predict(test_inputs, batch_size=32, verbose=1)

threshold = config.THRESHOLD

pred = np.where(predictions>threshold, 1,0)
def decode_tweet(original_tweet,idx_start,idx_end,offsets):

    filtered_output  = ""

    for ix in range(idx_start, idx_end + 1):

        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]

        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:

            filtered_output += " "

    return filtered_output
outputs = []

for test_idx in range(test.shape[0]):

    indexes = list(np.where(pred[test_idx]==1)[0])

    text = str(test.loc[test_idx,'text'])

    encoded_text = config.TOKENIZER.encode(text)

    if len(indexes)>0:

        start = indexes[0]

        end =  indexes[-1]

    else:  #if nothing was found above threshold value

        start = 0

        end = len(encoded_text.ids) - 1

    if end >= len(encoded_text.ids):

        end = len(encoded_text.ids) - 1

    if start>end: 

        selected_text = test.loc[test_idx,'text']

    else:

        selected_text = decode_tweet(text,start,end,encoded_text.offsets)

    outputs.append(selected_text)

    

test['selected_text'] = outputs
def replacer(row):

    if row['sentiment'] == 'neutral' or len(row['text'].split())<2:

        return row['text']

    else:

        return row['selected_text']

test['selected_text'] = test.apply(replacer,axis=1)
submission['selected_text'] = test['selected_text']

submission.to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 80)
submission.head()