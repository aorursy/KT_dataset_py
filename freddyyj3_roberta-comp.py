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
max_seq_length = 128  # Your choice here.

import tensorflow_hub as hub

import tensorflow as tf

import math

import transformers

from sklearn.model_selection import StratifiedKFold

import re

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = transformers.RobertaTokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)
train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

train.head(10)
#train.set_index('textID')

#test.set_index('textID')

train['text']=train['text'].fillna(" ")

test['text']=test['text'].fillna(" ")

train['selected_text']=train['selected_text'].fillna(" ")
stop_words = set(stopwords.words('english'))

for i in range(len(train['selected_text'])):

    text=train['selected_text'][i]

    text=text.strip()

#     text=text.lower()

#     text=re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

#     text=' '.join(re.sub("(@[A-Za-z0-9]+)"," ",text).split())

    train['selected_text'][i]=text

train.head(10)
train_x = train['text'].tolist()

# train_x = np.array(train_x, dtype=object)[:, np.newaxis]

train_y = train['selected_text'].tolist()



test_x = test['text'].tolist()

# test_x = np.array(test_x, dtype=object)[:, np.newaxis]

# test_y = test['selected_text'].tolist()
# for i in range(len(train_y)):

#     train_y[i]=train_y[i].strip()
sentiment=train['sentiment'].tolist()
def map_example_to_dict(input_ids, attention_masks,type_ids):

  return [

      tf.convert_to_tensor(input_ids),

      tf.convert_to_tensor(attention_masks),

      tf.convert_to_tensor(type_ids)

  ]
train_token=[]

test_token=[]

for i in range(len(train_x)):

    train_token.append(tokenizer.encode_plus(train_x[i],pad_to_max_length=True,max_length=max_seq_length,return_token_type_ids=True))

for i in range(len(test_x)):

    test_token.append(tokenizer.encode_plus(test_x[i],pad_to_max_length=True,max_length=max_seq_length,return_token_type_ids=True))

print(train_token[0])
input_ids=[]

attention_mask=[]

type_ids=[]

for i in train_token:

    input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))

    attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))

    type_ids.append(tf.reshape(i['token_type_ids'],(-1,max_seq_length)))

train_input=map_example_to_dict(input_ids,attention_mask,type_ids)

print(len(train_input[0]))

#print(len(train_input[0][0]))
input_ids=[]

attention_mask=[]

type_ids=[]

for i in test_token:

    input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))

    attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))

    type_ids.append(tf.reshape(i['token_type_ids'],(-1,max_seq_length)))

test_input=map_example_to_dict(input_ids,attention_mask,type_ids)

print(len(test_input[0]))

#print(len(test_input[0][0]))
ids = train_input[0]

masks = train_input[1]

token_ids=train_input[2]



ids = tf.reshape(ids, (-1, max_seq_length,))

print("Input ids shape: ", ids.shape)

masks = tf.reshape(masks, (-1, max_seq_length,))

print("Input Masks shape: ", masks.shape)

token_ids = tf.reshape(token_ids, (-1, max_seq_length,))

print("Token Ids shape: ", token_ids.shape)



ids=ids.numpy()

masks = masks.numpy()

token_ids=token_ids.numpy()
test_ids = test_input[0]

test_masks = test_input[1]

test_token_ids=test_input[2]



test_ids = tf.reshape(test_ids, (-1, max_seq_length,))

print("Input ids shape: ", test_ids.shape)

test_masks = tf.reshape(test_masks, (-1, max_seq_length,))

print("Input Masks shape: ", test_masks.shape)

test_token_ids = tf.reshape(test_token_ids, (-1, max_seq_length,))

print("Token Ids shape: ", test_token_ids.shape)



test_ids=test_ids.numpy()

test_masks = test_masks.numpy()

test_token_ids=test_token_ids.numpy()
ct = train.shape[0]

start_tokens = np.zeros((ct,max_seq_length),dtype='int32')

end_tokens = np.zeros((ct,max_seq_length),dtype='int32')



for k in range(train.shape[0]):

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

            

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
config = transformers.RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

bert_model = transformers.TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),loss='categorical_crossentropy')
import pickle



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean(loss)

    return loss
def build_model():

    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)

    attention_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)

    token_type = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)

    padding = tf.cast(tf.equal(ids, 1), tf.int32)

    

    lens = max_seq_length - tf.reduce_sum(padding, -1)

    max_len_ = tf.reduce_max(lens)

    ids_ = input_ids[:, :max_len_]

    att_ = attention_mask[:, :max_len_]

    tok_ = token_type[:, :max_len_]

    bert_layer = bert_model(ids_, attention_mask=att_,token_type_ids=tok_)[0]



    x1 = tf.keras.layers.Dropout(0.1)(bert_layer) 

    x1 = tf.keras.layers.Conv1D(768,2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)



    x2 = tf.keras.layers.Dropout(0.1)(bert_layer) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.Model(inputs=[input_ids, attention_mask,token_type], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    x1_padded = tf.pad(x1, [[0, 0], [0, max_seq_length - max_len_]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, max_seq_length - max_len_]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[input_ids, attention_mask, token_type], outputs=[x1_padded,x2_padded])

    return model,padded_model
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0, patience=3)
jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((ids.shape[0],max_seq_length))

oof_end = np.zeros((ids.shape[0],max_seq_length))

preds_start = np.zeros((test_ids.shape[0],max_seq_length))

preds_end = np.zeros((test_ids.shape[0],max_seq_length))
EPOCHS = 3 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

SEED = 88888

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)
import random



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=random.randint(100,999))

for fold,(idxT,idxV) in enumerate(skf.split(ids,sentiment)):

    

    tf.keras.backend.clear_session()

    model,padded_model = build_model()

    sv = tf.keras.callbacks.ModelCheckpoint(

        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

        save_weights_only=True, mode='auto', save_freq='epoch')

    

    inpT = [ids[idxT,], masks[idxT,], token_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [ids[idxV,],masks[idxV,],token_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == 1).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '../input/roberta2conv1dmodel/%s-roberta-%i.h5'%(VER,fold)

    #weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

#         model.fit(inpT, targetT, 

#             epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY,

#             validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

#         save_weights(model, weight_fn)

    

    load_weights(model, weight_fn)

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([ids[idxV,],masks[idxV,],token_ids[idxV,]],verbose=DISPLAY)

    preds = padded_model.predict([test_ids,test_masks,test_token_ids],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits



    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc[a-1:])

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc[a-1:b])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL CV Jaccard =',np.mean(jac))
all = []

for k in range(test_ids.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        #st = test.loc[k,'text']

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc[b:a])

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        

        st = tokenizer.decode(enc[a:b])

        

    all.append(st)
result=[]

for i in all:

    result.append("%s"%i)

for i in range(len(result)):

    text=result[i]

    text=text.lower()

    text=re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

    text=' '.join(re.sub("(@[A-Za-z0-9]+)"," ",text).split())

    tokenized=word_tokenize(text)

    preprocessed=[]

#     for j in tokenized:

#         if j not in stop_words:

#             preprocessed.append(j)

#     text=' '.join(preprocessed)

#    text=remove_all_consecutive(text)

    alphanumeric=""

    for character in text:

        if character not in ["`",",",'(',')','!',"'",'.','?']:

            alphanumeric += character

        else:

            alphanumeric+=' '

#     text=alphanumeric

    text=text.strip()

    result[i]=text



#     #text=''.join(i for i in text if (not i.isdigit() or i==" "))
test['selected_text'] = result

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)