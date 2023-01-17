SEED = 88888

import os
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import transformers
from transformers.optimization_tf import AdamWeightDecay
import tokenizers
from functools import partialmethod
import pandas as pd
import math
import re 

print('TF version',tf.__version__)
MAX_LEN = 96
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)
EPOCHS = 100 # originally 3
BATCH_SIZE = 32 # originally 32
PAD_ID = 1
LABEL_SMOOTHING = 0.15

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
#train.head()
ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
pre_enc = 4
post_enc = 1
tot_enc = pre_enc+post_enc

for k in range(train.shape[0]):    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    idx = (text1 + " ").find(" "+ text2 + " ")
    if idx < 0:
        idx = text1.find(text2)
    else:
        idx += 1
    
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
    input_ids[k,:len(enc.ids)+ tot_enc] = [0,s_tok,s_tok,2] + enc.ids + [2]
    attention_mask[k,:len(enc.ids)+ tot_enc] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+ pre_enc ] = 1
        end_tokens[k,toks[-1]+ pre_enc] = 1
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test.shape[0]):        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    if len(enc.ids) > MAX_LEN - tot_enc:
        continue
    input_ids_t[k,:len(enc.ids)+ tot_enc] = [0,s_tok,s_tok,2] + enc.ids + [2]
    attention_mask_t[k,:len(enc.ids)+ tot_enc] = 1
import pickle

def create_optimizer(
    init_lr,
    num_train_steps,
    num_warmup_steps,
    min_lr_ratio=0.0,
    adam_epsilon=1e-8,
    weight_decay_rate=0.0,
    include_in_weight_decay=None,
):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=init_lr * min_lr_ratio,
    )
    if num_warmup_steps:
        lr_schedule = transformers.optimization_tf.WarmUp(
            initial_learning_rate=init_lr, decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps
        )
    if weight_decay_rate > 0.0:
        AdamWeightDecay.apply_gradients = partialmethod(AdamWeightDecay.apply_gradients, clip_norm=1.0)
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            include_in_weight_decay=include_in_weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=adam_epsilon)
    # We return the optimizer and the LR scheduler in order to better track the
    # evolution of the LR independently of the optimizer.
    return optimizer, lr_schedule

def scheduler(epoch):
    if epoch > 3:
        return 3e-5 * 0.5**(3)
    return 3e-5 * 0.5**(epoch)
    
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
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)
    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)
    
    x1 = tf.keras.layers.Dropout(0.5)(x[0])
    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.4)(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.5)(x[0]) 
    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.4)(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    lr_schedule = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=3e-5, \
                                                        first_decay_steps=4550, t_mul=2.0, m_mul=0.85, alpha=0.1)
    AdamWeightDecay.apply_gradients = partialmethod(AdamWeightDecay.apply_gradients, clip_norm=1.0)
    test_optimizer = AdamWeightDecay(
                            learning_rate=lr_schedule,
                            weight_decay_rate=0.003,
                            include_in_weight_decay=None,
                        )
    model.compile(loss=loss_fn, optimizer=test_optimizer)
    
    # this is required as `model.predict` needs a fixed size!
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    
    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])
    return model, padded_model

def jaccard(str1, str2): 
    #print(str1,str2)
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0],MAX_LEN))
oof_end = np.zeros((input_ids.shape[0],MAX_LEN))
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
train["predicted"] = ""

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED) #originally 5 splits
for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)
    
    K.clear_session()
    model, padded_model = build_model()
    best_vall_loss = 10000
    no_loss_improvement=0
    patience = 20
     
    
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler,\
                                                         verbose=DISPLAY)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                       patience=1, verbose=DISPLAY, mode='auto')
    
    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]
    targetT = [start_tokens[idxT,], end_tokens[idxT,]]
    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]
    targetV = [start_tokens[idxV,], end_tokens[idxV,]]
    # sort the validation data
    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))
    inpV = [arr[shuffleV] for arr in inpV]
    targetV = [arr[shuffleV] for arr in targetV]
    weight_fn = '../models/test/%s-roberta-%i.h5'%(VER,fold)
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
        history = model.fit(inpT, targetT,\
                  epochs=epoch, initial_epoch=epoch - 1, \
                  batch_size=BATCH_SIZE, \
                  verbose=DISPLAY, callbacks=[],\
                  #steps_per_epoch = 30,\
                  validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`
        
        if (history.history['val_loss'][-1]<best_vall_loss): 
            no_loss_improvement = 0
            print("\nsaving as val loss reduced:",best_vall_loss," to ",history.history['val_loss'][-1])
            best_vall_loss = history.history['val_loss'][-1]
            save_weights(model, weight_fn)
        else :
            no_loss_improvement +=1
            if no_loss_improvement == patience:
                print("\nearly stopping")
                break
        
    print('Loading model...')
    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    load_weights(model, weight_fn)

    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],\
                                                            attention_mask[idxV,],\
                                                            token_type_ids[idxV,]],\
                                                           verbose=DISPLAY)
    
    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        text1 = " "+" ".join(train.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        if (a-pre_enc < 0): a= pre_enc
        if (b-pre_enc+1 < 0): b= pre_enc
        if a>b: 
            st = tokenizer.decode(enc.ids[a-pre_enc:]+enc.ids[:b-pre_enc+1])
            #st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here
        else:
            st = tokenizer.decode(enc.ids[a-pre_enc:b-pre_enc+1])
        all.append(jaccard(st,train.loc[k,'selected_text']))
        train.loc[k,"predicted"] = st
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))
    
    print('Predicting Test...')
    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/skf.n_splits * np.mean(all)
    preds_end += preds[1]/skf.n_splits * np.mean(all)
    
    print()
train["pred_jac"] = train.apply(lambda x: jaccard(x["selected_text"],x["predicted"]),axis=1)
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
print(jac) # Jaccard CVs
all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)
    if (a-pre_enc < 0): a= pre_enc
    if (b-pre_enc+1 < 0): b=pre_enc
    if a>b: 
        st = tokenizer.decode(enc.ids[a-pre_enc:]+enc.ids[:b-pre_enc+1])
        #st = test.loc[k,'text']
    else:
        st = tokenizer.decode(enc.ids[a-pre_enc:b-pre_enc+1])
    all.append(st)
test['selected_text'] = all

test['selected_text'] = test[['selected_text','text']].apply(\
                                                       lambda x: re.sub(r'\.\.+$', "..", x[0]) \
                                                       if (len(x[0].split())==1 and len(x[1].split())!=1) \
                                                       else x , axis =1)
test['selected_text'] = test[['selected_text','text']].apply(\
                                                       lambda x: re.sub(r'!!!!+$', "!", x[0]) \
                                                       if (len(x[0].split())==1 and len(x[1].split())!=1) \
                                                       else x , axis =1)

test[['textID','selected_text']].to_csv('submission.csv',index=False)
def replace(string): 
  
    # findall() has been used  
    # with valid conditions for urls in string 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)
    clean_str = re.sub(regex,'',string)
    return len(url),clean_str

def find_special_chars(string):
    if len(string.strip())==0:
        return ""
    if string.strip()[0] == "_":
        words = string.split()
        one_word = words[0][1:]+" "+words[0]+" "
        print(string)
        print(one_word + " ".join(words[1:]))
        return one_word +" ".join(words[1:])
    return string
    
def display_vals(x):  
    print("text=",x['text'])
    print("selected_text=",x['selected_text'])
    print("predicted=",x['predicted'])
    print("sentiment=",x['sentiment'])
    print("jaccard=",str(x['pred_jac']))
    print("test_pred=",str(x['pred_test']))
    print("pred_test_jac=",str(x['pred_test_jac']))
    print("\n")
#train.to_csv("train_pred.csv",index=False)
#train = pd.read_csv("./train_pred.csv").fillna('')
train["select_eq_text"] = train.apply(lambda x: jaccard(x["selected_text"],x["text"]),axis=1)
train.loc[train["text"]=='',"text"]=" "
train.loc[train["selected_text"]=='',"selected_text"]=" "
train["pred_test"] = train.predicted

train['pred_test'] = train[['pred_test','text']].apply(\
                                                       lambda x: re.sub(r'\.\.+$', "..", x[0]) \
                                                       if (len(x[0].split())==1 and len(x[1].split())!=1) \
                                                       else x , axis =1)
train['pred_test'] = train[['pred_test','text']].apply(\
                                                       lambda x: re.sub(r'!!!!+$', "!", x[0]) \
                                                       if (len(x[0].split())==1 and len(x[1].split())!=1) \
                                                       else x , axis =1)

#train['pred_test'] = train['pred_test'].apply(lambda x: re.sub(r',.*',",",x))

#train['pred_test'] = train['pred_test'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
#train['pred_test'] = train['pred_test'].apply(lambda x: find_special_chars(x))

train["pred_test_jac"] = train.apply(lambda x: jaccard(x["selected_text"],x["pred_test"]),axis=1)
print(train.pred_test_jac.mean(),train.pred_jac.mean())
print("overall gain =",(train["pred_test_jac"] - train["pred_jac"]).sum())