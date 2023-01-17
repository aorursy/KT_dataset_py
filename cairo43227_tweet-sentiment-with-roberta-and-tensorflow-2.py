# !pip uninstall tensorflow

# !pip install --ignore-installed --upgrade tensorflow-gpu



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

import tokenizers

from transformers import RobertaTokenizer, RobertaConfig, TFRobertaPreTrainedModel

from transformers.modeling_tf_roberta import TFRobertaMainLayer,TFRobertaModel

from transformers.modeling_tf_utils import get_initializer

import tensorflow.keras.backend as K



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    

print('TF version',tf.__version__)
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').fillna('')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv').fillna('')
train.head()
test.head()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(str(train.loc[k,'text']).split()) # TODO achar o float

    text2 = " ".join(str(train.loc[k,'selected_text']).split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc.ids)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1
def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids,attention_mask=att,token_type_ids=tok)



    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

#     x1 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=48,

#                 dropout=0.3,

#                 return_sequences=False

#             )

#         )(x1)

    x1 = tf.keras.layers.Flatten()(x1)

 

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

#     x2 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=48,

#                 dropout=0.3,

#                 return_sequences=False

#             )

#         )(x2) 

    x2 = tf.keras.layers.Flatten()(x2)

    

    x2 = tf.keras.layers.Activation('softmax')(x2)





#     x1 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=128,

#                 dropout=0.3,

#                 return_sequences=True

#             )

#         )(x[0])



#     x1 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=128,

#                 dropout=0.3,

#                 return_sequences=False

#             )

#         )(x1)    

#     x1 = tf.keras.layers.Dense(128, activation='relu')(x1)

#     x1 = tf.keras.layers.Dropout(0.3)(x1)

#     x1 = tf.keras.layers.Flatten()(x1)

#     x1 = tf.keras.layers.Dense(1, activation='softmax')(x1)

    

    

#     x2 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=128,

#                 dropout=0.3,

#                 return_sequences=True

#             )

#         )(x[0])



#     x2 = tf.keras.layers.Bidirectional(

#             tf.keras.layers.LSTM(

#                 units=128,

#                 dropout=0.3,

#                 return_sequences=False

#             )

#         )(x2)    

#     x2 = tf.keras.layers.Dense(128, activation='relu')(x2)

#     x2 = tf.keras.layers.Dropout(0.3)(x2)

#     x2 = tf.keras.layers.Flatten()(x2)

#     x2 = tf.keras.layers.Dense(1, activation='softmax')(x2)

    

    

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
def lr(epoch):

    return 3e-5 * 0.2**epoch



# schedule = tf.keras.callbacks.LearningRateScheduler(lr, verbose=2)
jac = []; VER='v0'; DISPLAY=2 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

FOLDS = 5



skf = StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=777)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model = build_model()

#     print(model.summary())

        

#     sv = tf.keras.callbacks.ModelCheckpoint(

#         '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

#         save_weights_only=True, mode='auto', save_freq='epoch')

 

#     model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 

#         epochs=5, batch_size=60, verbose=DISPLAY, callbacks=[sv], validation_data=([input_ids[idxV,],

#         attention_mask[idxV,],token_type_ids[idxV,]], [start_tokens[idxV,], end_tokens[idxV,]]))

    

    print('Loading model...')

    model.load_weights('/kaggle/input/robertaweights1/%s-roberta-%i.h5'%(VER,fold))#

    

    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],

                                                    verbose=DISPLAY)

    

    print('Predicting Test...')

    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-1:b])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)
test['selected_text'] = ['"{}"'.format(x) for x in all]

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)