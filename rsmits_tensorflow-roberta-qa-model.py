# Import Modules

import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *
# Show versions of Tensorflow

print('TF version',tf.__version__)
# Set other Seeds

SEED = 4262

np.random.seed(SEED)

tf.random.set_seed(SEED)
# Set strategy for tpu

USE_TPU = False

if USE_TPU:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.OneDeviceStrategy(device = "/gpu:0")
# Constants

MAX_LEN = 128

FOLDS = 5

EPOCHS = 3

VERBOSE = 1

ROBERTA_BASE_PATH = '../input/tf-roberta/'



BATCH_SIZE = 32 * strategy.num_replicas_in_sync

print('BatchSize: {}'.format(BATCH_SIZE))



LR = 3e-5 * strategy.num_replicas_in_sync

print('LearningRate: {}'.format(LR))



# Set the following to True after training to make a submission

INFERENCE = True
# Read Train Data (I read in a multiple of batch size 256 when training on TPU)

#train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', nrows = 26880).fillna('')

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')



# Summary

train.head()
# Read Test Data

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



# Summary

test.head()
# Tokenizer

tokenizer = RobertaTokenizer(vocab_file = ROBERTA_BASE_PATH + 'vocab-roberta-base.json',

                             merges_file = ROBERTA_BASE_PATH + 'merges-roberta-base.txt',

                             add_prefix_space = True,

                             do_lower_case = True)
# Pre Process Training Data

ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN), dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN), dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN), dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN), dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN), dtype='int32')



for k in range(ct):

    # Process Text

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    

    # Skip rows where Text1 is empty (RobertaTokenizer crashes: https://github.com/huggingface/transformers/issues/3809)

    if text1 != '':

        # Encode Full input sample

        input_encoded = tokenizer.encode_plus(text1, train.loc[k,'sentiment'], add_special_tokens = True, max_length = MAX_LEN)

        input_ids_sample = input_encoded["input_ids"]

        attention_mask_sample = input_encoded["attention_mask"]



        # Attention Mask

        attention_mask[k,:len(attention_mask_sample)] = attention_mask_sample



        # Input Ids

        input_ids[k,:len(input_ids_sample)] = input_ids_sample



        # Find overlap between Full Text and Selected Text

        idx = text1.find(text2)

        chars = np.zeros((len(text1)))

        chars[idx:idx + len(text2)] = 1

        k_ids = tokenizer.encode(text1, add_special_tokens = False) 



        # ID_OFFSETS

        offsets = [] 

        idx = 0

        for t in k_ids:

            w = tokenizer.decode([t])

            offsets.append((idx,idx+len(w)))

            idx += len(w)



        # Get Start and End Tokens

        toks = []

        for i,(a,b) in enumerate(offsets):

            sm = np.sum(chars[a:b])

            if sm>0: 

                toks.append(i) 

        if len(toks) > 0:

            start_tokens[k,toks[0]+1] = 1

            end_tokens[k,toks[-1]+1] = 1
# Pre Process Test Data

ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(ct):

    # Process Text

    text1 = " "+" ".join(test.loc[k,'text'].split())

    

    # Skip rows where Text1 is empty (RobertaTokenizer crashes: https://github.com/huggingface/transformers/issues/3809)

    if text1 != '':

        # Encode Full input sample

        input_encoded = tokenizer.encode_plus(text1, test.loc[k, 'sentiment'], add_special_tokens = True, max_length = MAX_LEN)

        input_ids_sample = input_encoded["input_ids"]

        attention_mask_sample = input_encoded["attention_mask"]



        # Attention Mask

        attention_mask_t[k,:len(attention_mask_sample)] = attention_mask_sample



        # Input Ids

        input_ids_t[k,:len(input_ids_sample)] = input_ids_sample
# Metric

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    

    if (len(a)==0) & (len(b)==0): 

        return 0.5

    

    c = a.intersection(b)

    

    return float(len(c)) / (len(a) + len(b) - len(c))
def custom_loss(y_true, y_pred):

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits = False, label_smoothing = 0.1)

    loss = tf.reduce_mean(loss)

    

    return loss
# Config

config = RobertaConfig.from_pretrained(ROBERTA_BASE_PATH+'config-roberta-base.json')

#config.attention_probs_dropout_prob = 0.15

print(config)
def build_model():

    # Create Model

    with strategy.scope():      

        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)



        roberta_model = TFRobertaForQuestionAnswering.from_pretrained(ROBERTA_BASE_PATH + 'pretrained-roberta-base.h5', config = config)

        x = roberta_model(ids, attention_mask = att, token_type_ids = tok)



        x1 = tf.keras.layers.Dropout(0.30)(x[0]) 

        x1 = tf.keras.layers.Activation('softmax')(x1)



        x2 = tf.keras.layers.Dropout(0.30)(x[1]) 

        x2 = tf.keras.layers.Activation('softmax')(x2)



        model = tf.keras.models.Model(inputs = [ids, att, tok], outputs=[x1, x2])

        optimizer = tf.keras.optimizers.Adam(learning_rate = LR)



        model.compile(loss = custom_loss, optimizer = optimizer)



        return model
# Placeholders

jac, jac1 = [], []

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits = FOLDS, shuffle = True, random_state = SEED)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    # Clear session and create Model

    K.clear_session()

    model = build_model()

        

    # Callbacks

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('roberta-%i.h5'%(fold), monitor = 'val_loss', verbose = 1, save_best_only = True,

                                                          save_weights_only = True, mode = 'auto', save_freq = 'epoch')

    

    if not INFERENCE:

        # Train Model 

        model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 

                  epochs = EPOCHS, 

                  batch_size = BATCH_SIZE, 

                  verbose = VERBOSE, 

                  callbacks = [model_checkpoint],

                  validation_data = ([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], [start_tokens[idxV,], end_tokens[idxV,]]),

                  shuffle = True)



        print('Loading model...')

        model.load_weights(f'roberta-{fold}.h5')        

    else:

        print('Loading model...')

        model.load_weights(f'/kaggle/input/tensorflow-roberta-qa-model/roberta-{fold}.h5')

    

    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose = VERBOSE)

    

    print('Predicting Test...')

    preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose = VERBOSE)

    preds_start += preds[0] / skf.n_splits

    preds_end += preds[1] / skf.n_splits

    

    # Display Fold Jaccard

    all = []

    all1 = []



    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        

        # Encode Text

        text1 = " "+" ".join(train.loc[k,'text'].split())

        enc = tokenizer.encode(text1, add_special_tokens = False)

        

        if train.loc[k,'sentiment'] == 'neutral':

            st = " ".join(train.loc[k,'text'].split())

            st1 = " ".join(train.loc[k,'text'].split())

        else:

            # Check if start comes after end

            if a > b:

                st = text1

                st1 = text1



                # Sort according to max probabilities and get the indices            

                start_sort = np.argsort(oof_start[k,])[::-1] 

                end_sort = np.argsort(oof_end[k,])[::-1]



                a1 = start_sort[1]

                b1 = end_sort[1]

                a2 = start_sort[2]

                b2 = end_sort[2]



                # Try the next 2 positions..if one of them has the correct order

                if a1 < b1:

                    st1 = tokenizer.decode(enc[a1-1:b1])

                elif a2 < b2:

                    st1 = tokenizer.decode(enc[a2-1:b2])   

            else:

                st = tokenizer.decode(enc[a-1:b])

                st1 = tokenizer.decode(enc[a-1:b])



        # Store Results

        all.append(jaccard(st, train.loc[k,'selected_text']))

        all1.append(jaccard(st1, train.loc[k,'selected_text']))



    jac.append(np.mean(all))

    jac1.append(np.mean(all1))

    

    print('>>>> FOLD %i Jaccard ='%(fold+1), np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1), np.mean(all1))

    

    print()
print(f'=== OVERALL 5 Fold CV Jaccard                  = {np.mean(jac)}')

print(f'=== OVERALL 5 Fold CV Jaccard - Post Processed = {np.mean(jac1)}')
# Generate final results for submission

all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    

    # Encode Text

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1, add_special_tokens = False)

    

    if test.loc[k, 'sentiment'] == 'neutral':

        st = " ".join(test.loc[k,'text'].split())

    else:

        # Check if start comes after end

        if a > b:  

            st = text1



            # Sort according to max probabilities and get the indices            

            start_sort = np.argsort(preds_start[k,])[::-1] 

            end_sort = np.argsort(preds_end[k,])[::-1]



            a1 = start_sort[1]

            b1 = end_sort[1]

            a2 = start_sort[2]

            b2 = end_sort[2]



            # Try the next 2 positions..if one of them has the correct order

            if a1 < b1:

                st = tokenizer.decode(enc[a1-1:b1])

            elif a2 < b2:

                st = tokenizer.decode(enc[a2-1:b2])  

        else:

            st = tokenizer.decode(enc[a-1:b])



    all.append(st)
# Store Submission and show some results

test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)