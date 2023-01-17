# importing libraries



import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import *

import tokenizers

print('TF version',tf.__version__)

import os

from tokenizers import BertWordPieceTokenizer

from transformers import BertTokenizer, TFBertModel, BertConfig

from sklearn.model_selection import StratifiedShuffleSplit

from transformers import BertTokenizer, TFBertForQuestionAnswering

from sklearn.model_selection import train_test_split

import gc

from keras.callbacks import ModelCheckpoint

max_len = 128

train_mode = False



# Load the fast tokenizer from saved file

tokenizer = BertWordPieceTokenizer("../input/bert-qa-best/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt", lowercase=True)

    
# loading train data.

sentiment_id = {'positive': 3893, 'negative': 4997, 'neutral': 8699}

data = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')

data.head()
# removing empty rows

data['text'].replace('', np.nan, inplace=True)

data.dropna(subset=['text'], inplace=True)

data.reset_index(drop=True, inplace=True)
x_train,x_test = train_test_split(data, test_size = 0.05, random_state=42)

x_train,x_cv = train_test_split(x_train, test_size = 0.2, random_state = 42)



print("x_train shape is", x_train.shape)

print("x_cv shape is", x_cv.shape)

print("x_test shape is", x_test.shape)

x_train.reset_index(drop=True, inplace=True)

x_cv.reset_index(drop=True, inplace=True)

x_test.reset_index(drop=True, inplace=True)
# https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch

# Example to explain operations in text_process function



tweet = 'WHY DO WE FALL? SO THAT WE CAN LEARN TO PICK OURSELVES BACK UP.'

selected_text = 'LEARN TO PICK OURSELVES BACK UP'

sentiment = 'positive'

print('text',tweet)

print('select_text:',selected_text)

print('sentiment:', sentiment)

# idx0 and idx1 and start and end indices of select_text in tweet.

idx0 = None

idx1 = None

st_len = len(selected_text)

for i in range(len(tweet)):

    if(tweet[i:i+st_len]==selected_text):

        idx0 = i

        idx1 = i + st_len -1

        break



# char_tartgets is of length tweet, assign indices with select text =1 

char_targets = [0]*len(tweet)

for i in range(len(tweet)):

    if idx0 != None and idx1!=None:

        if i>=idx0 and i<=idx1:

            char_targets[i] = 1



print('char_targets:',char_targets)

# encoding tweet using tokenizer, it returns ids(token for each word) and offsets(span of each word)

tok_tweet = tokenizer.encode(tweet)



input_ids = tok_tweet.ids[1:-1] # word ids given by tokenizer stripping first[cls] and last token [sep]

offsets = tok_tweet.offsets[1:-1] # offsets of the tweet 



print('input_ids:',input_ids)

print('offsets:',offsets)

# start index and end index of tweet words with select_text

targets_index = []

for i, (off1,off2) in enumerate(offsets):

    if sum(char_targets[off1:off2])>0:

        targets_index.append(i)       

target_start = targets_index[0] 

target_end = targets_index[-1]



print('target_start:',target_start)

print('target_end:', target_end)



# creating ids, token_type_ids, mask into bert format, changing target_start and target_end accordingly.

ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids  + [102]

token_type_ids = [0,0,0] + [1]*(len(input_ids) + 1)

mask = [1] * len(token_type_ids)

target_start+=3 

target_end+=3

offsets = [(0,0)]*3 + offsets + [(0,0)]



# padding 

padding_length = max_len - len(ids)

if padding_length > 0:

    ids = ids + ([0] * padding_length)

    mask = mask + ([0] * padding_length)

    token_type_ids = token_type_ids + ([0] * padding_length)

    offsets = offsets + ([(0, 0)] * padding_length)

# https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch



def text_process(tweet, sentiment, tokenizer, max_len, selected_text=None):

    

    """

    inputs:

    tweets: text 

    sentiment: sentiment of the tweet

    tokenizer: tokenizer

    max_len: max length of ids, mask and token_type_ids (inputs of bert)

    selected_text: selected_text (optional)

    

    operation:

    

    Given inputs it calculates  ids, mask , token_type_ids, offsets and target_start and target_end.

    

    outputs:

    dictionary with keys as below,

    ids: input tokens for bert in format as 101 <sentiment tokens> 102 <text tokens> 102

    mask: array with length as max_len and has 1's in the indices of text and zeros elsewhere.

    token_type_ids: 1's in the place of text and zeros elsewhere , size max_len

    target_start,target_end: begin and end of select_text (returned only when select_text is given)

    offsets: offsets of text 

    

    """

    

    if selected_text!=None:

        idx0 = None

        idx1 = None

        st_len = len(selected_text)

        for i in range(len(tweet)):

            if(tweet[i:i+st_len]==selected_text):

                idx0 = i

                idx1 = i + st_len -1

                break



        char_targets = [0]*len(tweet)



        for i in range(len(tweet)):

            if idx0 != None and idx1!=None:

                if i>=idx0 and i<=idx1:

                    char_targets[i] = 1



        tok_tweet = tokenizer.encode(tweet)



        input_ids = tok_tweet.ids[1:-1] 

        offsets = tok_tweet.offsets[1:-1] 

        

        targets_index = []



        for i, (off1,off2) in enumerate(offsets):

            if sum(char_targets[off1:off2])>0:

                targets_index.append(i)



        target_start = targets_index[0] 

        target_end = targets_index[-1]

        

        

        ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids  + [102]

        token_type_ids = [0,0,0] + [1]*(len(input_ids) +1)

        mask = [1] * len(token_type_ids)

        target_start+=3 

        target_end+=3

        offsets = [(0,0)]*3 + offsets + [(0,0)]



        padding_length = max_len - len(ids)

        if padding_length > 0:

            ids = ids + ([0] * padding_length)

            mask = mask + ([0] * padding_length)

            token_type_ids = token_type_ids + ([0] * padding_length)

            offsets = offsets + ([(0, 0)] * padding_length)

            

        return {

            

            'ids': ids,

            'token_type_ids':token_type_ids,

            'mask':mask,

            'target_start':target_start,

            'target_end':target_end,

            'offsets':offsets

        }

    else:



        tok_tweet = tokenizer.encode(tweet)

        

        input_ids = tok_tweet.ids[1:-1] 

        offsets = tok_tweet.offsets[1:-1] 



        ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids  + [102]

        token_type_ids = [0,0,0] + [1]*(len(input_ids) + 1)

        mask = [1] * len(token_type_ids)

        offsets = [(0,0)]*3 + offsets + [(0,0)]



        padding_length = max_len - len(ids)

        if padding_length > 0:

            ids = ids + ([0] * padding_length)

            mask = mask + ([0] * padding_length)

            token_type_ids = token_type_ids + ([0] * padding_length)

            offsets = offsets + ([(0, 0)] * padding_length)

            

        return {

            

            'ids': ids,

            'token_type_ids':token_type_ids,

            'mask':mask,

            'offsets':offsets

        }



rows = x_train.shape[0]



train_ids = np.zeros((rows,max_len), dtype = 'int32')

train_mask = np.zeros((rows,max_len), dtype = 'int32')

train_type_ids = np.zeros((rows,max_len), dtype = 'int32')

train_start_idx = np.zeros((rows,max_len), dtype = 'int32')

train_end_idx = np.zeros((rows,max_len), dtype = 'int32')



for i in range(x_train.shape[0]):

    

    encoding = text_process(x_train.loc[i,'text'], x_train.loc[i,'sentiment'], tokenizer, max_len,x_train.loc[i,'selected_text'] )

    

    train_ids[i] = encoding['ids']

    train_start_idx[i,encoding['target_start']] = 1

    train_end_idx[i, encoding['target_end']] = 1

    

    train_type_ids[i] = encoding['token_type_ids']

    train_mask[i] = encoding['mask']
# checking



i = 10



encoding = text_process(x_train.loc[i,'text'], x_train.loc[i,'sentiment'], tokenizer, max_len,x_train.loc[i,'selected_text'] )



train_ids[i] = encoding['ids']





train_start_idx[i,encoding['target_start']] = 1

train_end_idx[i, encoding['target_end']] = 1



train_type_ids[i] = encoding['token_type_ids']

train_mask[i] = encoding['mask']



tweet = x_train.loc[i,'text']

select_text = x_train.loc[i,'selected_text']



target_start = np.argmax(train_start_idx[i,])

target_end = np.argmax(train_end_idx[i, ])



offsets = encoding['offsets']





print('tweet:',tweet)

print('selected_text:',select_text)

tweet[offsets[target_start][0]:offsets[target_end][-1]]
rows = x_cv.shape[0]



cv_ids = np.zeros((rows,max_len), dtype = 'int32')

cv_mask = np.zeros((rows,max_len), dtype = 'int32')

cv_type_ids = np.zeros((rows,max_len), dtype = 'int32')

cv_start_idx = np.zeros((rows,max_len), dtype = 'int32')

cv_end_idx = np.zeros((rows,max_len), dtype = 'int32')



for i in range(x_cv.shape[0]):

    

    encoding = text_process(x_cv.loc[i,'text'], x_cv.loc[i,'sentiment'], tokenizer, max_len,x_cv.loc[i,'selected_text'] )

    

    cv_ids[i] = encoding['ids']

    cv_start_idx[i,encoding['target_start']] = 1

    cv_end_idx[i, encoding['target_end']] = 1

    

    cv_type_ids[i] = encoding['token_type_ids']

    cv_mask[i] = encoding['mask']
rows = x_test.shape[0]



test_ids = np.zeros((rows,max_len), dtype = 'int32')

test_mask = np.zeros((rows,max_len), dtype = 'int32')

test_type_ids = np.zeros((rows,max_len), dtype = 'int32')

test_start_idx = np.zeros((rows,max_len), dtype = 'int32')

test_end_idx = np.zeros((rows,max_len), dtype = 'int32')



for i in range(x_test.shape[0]):

    

    encoding = text_process(x_test.loc[i,'text'], x_test.loc[i,'sentiment'], tokenizer, max_len,x_test.loc[i,'selected_text'] )

    

    test_ids[i] = encoding['ids']

    test_start_idx[i,encoding['target_start']] = 1

    test_end_idx[i, encoding['target_end']] = 1

    

    test_type_ids[i] = encoding['token_type_ids']

    test_mask[i] = encoding['mask']
# Metric

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    

    if (len(a)==0) & (len(b)==0): 

        return 0.5

    

    c = a.intersection(b)

    

    return float(len(c)) / (len(a) + len(b) - len(c))
def build_model():

    # Create Model

       

    ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    att = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    tok = tf.keras.layers.Input((max_len,), dtype=tf.int32)





    #bert = TFBertForQuestionAnswering.from_pretrained(modelName) this needs internet hence loading model from disk.

    bert = TFBertForQuestionAnswering.from_pretrained('../input/bert-squad/bert-large-uncased-whole-word-masking-finetuned-squad-tf_model.h5', config = '../input/bert-squad/bert-large-uncased-whole-word-masking-finetuned-squad-config.json')

    x = bert(ids, attention_mask = att, token_type_ids = tok)



    x1 = tf.keras.layers.Dropout(0.3)(x[0]) 

    x1 = tf.keras.layers.Activation('softmax')(x1)



    x2 = tf.keras.layers.Dropout(0.3)(x[1]) 

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs = [ids, att, tok], outputs=[x1, x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5)



   # model.compile(loss = custom_loss, optimizer = optimizer)

    model.compile(loss= 'categorical_crossentropy', optimizer=optimizer)

    return model
filepath = "/kaggle/working/best_model.h5" 

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True, mode = 'auto', save_freq = 'epoch')

model = build_model()





model.summary()
# clearing space

gc.collect()
# Training or loading trained model for predictions.

if train_mode:

    model.fit([train_input_ids, train_attention_mask, train_token_type_ids], [train_start_tokens, train_end_tokens], 

                      epochs = 3, 

                      batch_size = 8, 

                      verbose = VERBOSE, 

                      callbacks = [checkpoint],

                      validation_data = ([cv_input_ids,cv_attention_mask,cv_token_type_ids], [cv_start_tokens, cv_end_tokens]),

                      shuffle = True)

else:

    

    model.load_weights('../input/bert-qa-best/best_model (1).h5')
rows = x_test.shape[0]



preds_start = np.zeros((rows,max_len))

preds_end = np.zeros((rows,max_len))
preds = model.predict([test_ids, test_mask, test_type_ids], verbose = True)

preds_start += preds[0]

preds_end += preds[1] 


score = 0

for k in range(x_test.shape[0]):

    

        

    

        encoding = text_process(x_test.loc[k,'text'], x_test.loc[k,'sentiment'], tokenizer, max_len)

        offsets = encoding['offsets']

        #targets_start,targets_end = model.predict(encoding['ids'], encoding['mask'], encoding['token_type_ids'] )

        targets_start = np.argmax(preds_start[k,])

        targets_end = np.argmax(preds_end[k,])



        pred = x_test.loc[k,'text'][offsets[targets_start][0]:offsets[targets_end][-1]]

        score+=jaccard(x_test.loc[k,'selected_text'], pred)



    

    

score=score/x_test.shape[0]    

print('score on local test_data',score)
# prediction samples



for k in range(0,x_test.shape[0],100):

    

        encoding = text_process(x_test.loc[k,'text'], x_test.loc[k,'sentiment'], tokenizer, max_len)

        offsets = encoding['offsets']

        targets_start = np.argmax(preds_start[k,])

        targets_end = np.argmax(preds_end[k,])



        pred = x_test.loc[k,'text'][offsets[targets_start][0]:offsets[targets_end][-1]]

        print('text:', x_test.text[k])

        print('selected text:', x_test.selected_text[k])

        print('sentiment:',x_test.sentiment[k] )

        print('predicted:', pred)

        print('jaccard_score:', jaccard(pred, x_test.loc[k,'selected_text']))

        print('#########################################')

    
x_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

rows = x_test.shape[0]

preds_start = np.zeros((rows,max_len))

preds_end = np.zeros((rows,max_len))
rows = x_test.shape[0]



test_ids = np.zeros((rows,max_len), dtype = 'int32')

test_mask = np.zeros((rows,max_len), dtype = 'int32')

test_type_ids = np.zeros((rows,max_len), dtype = 'int32')

test_start_idx = np.zeros((rows,max_len), dtype = 'int32')

test_end_idx = np.zeros((rows,max_len), dtype = 'int32')



for i in range(x_test.shape[0]):

    

    encoding = text_process(x_test.loc[i,'text'], x_test.loc[i,'sentiment'], tokenizer, max_len )

    test_ids[i] = encoding['ids']

    test_type_ids[i] = encoding['token_type_ids']

    test_mask[i] = encoding['mask']

preds = model.predict([test_ids, test_mask, test_type_ids], verbose = True)

preds_start += preds[0]

preds_end += preds[1] 
submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')


for k in range(x_test.shape[0]):

    

    encoding = text_process(x_test.loc[k,'text'], x_test.loc[k,'sentiment'], tokenizer, max_len)

    offsets = encoding['offsets']

    targets_start = np.argmax(preds_start[k,])

    targets_end = np.argmax(preds_end[k,])



    pred = x_test.loc[k,'text'][offsets[targets_start][0]:offsets[targets_end][-1]]

    submission.loc[k,'selected_text'] = pred



submission.to_csv('submission.csv', index = False)