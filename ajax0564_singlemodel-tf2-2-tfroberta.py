import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

import math



from tqdm import tqdm

tqdm.pandas()



MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

EPOCHS = 3 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

SEED = 88888

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

train.head()
train.isnull().sum()
train = train.dropna(axis = 0)

train.isnull().sum()
def process(text, selected_text, ib_space):



    added_extra_space = False

    splitted = text.split(selected_text)

    if splitted[0][-1] == ' ':

        added_extra_space = True

        splitted = text.split(" " + selected_text)

    

    sub = len(splitted[0]) - len(" ".join(splitted[0].split()))

    if sub == 1 and text[0] == ' ':

        splitted = text.split(selected_text)

        add_space = True if splitted[0] and splitted[0][-1] == ' ' else False



        add_space = False

        if splitted[0] and splitted[0][-1] == ' ':

            add_space = True

        if add_space == False:

            start = text.find(selected_text)

        else:        

            start = text.find(selected_text) - 1

    elif sub == 2:

        start = text.find(selected_text)

    else:

        start = text.find(selected_text)

        



    splitted = text.split(selected_text)



    add_space = False

    if splitted[0] and splitted[0][-1] == ' ':

        add_space = True



    text_pr =  " ".join(splitted[0].split())

    if add_space:

        text_pr += " "



    text_pr = text_pr + selected_text



    if len(splitted) > 1:

        text_pr = text_pr + splitted[1]



    if sub > 1 :

        if text[0] == ' ': 

            end = start + len(selected_text) - 1 + ib_space            

        else:

            end = start + len(selected_text)

    else:

        end = start + len(selected_text)

        

    new_st = text_pr[start : end]

    return new_st

def process_selected_text(text, selected_text):

    splitted = text.split(selected_text)

    add_space = True if splitted[0] and splitted[0][-1] == ' ' else False    

    sub = len(splitted[0]) - len( " ".join(splitted[0].split()) )

    in_between_space = len(selected_text) - len( " ".join(selected_text.split()) )

        

    new_selected_text = selected_text

    if sub > 0 and text.strip() != selected_text.strip() and add_space == False and text.find(selected_text) != 0:

        if in_between_space == 0:

            new_selected_text = process(text, selected_text, in_between_space)

    return new_selected_text
train['new_selected_text'] = train.selected_text

train = train[train.textID != '12f21c8f19']

train['new_selected_text'] = train.progress_apply(lambda x: process_selected_text(x.text, x.selected_text), axis=1)
train.to_csv('new_train.csv',index = False)
train = pd.read_csv('/kaggle/working/new_train.csv')

train.head(5)
# ct = train.shape[0]

# input_ids = np.ones((ct,MAX_LEN),dtype='int32')

# attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

# token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

# start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

# end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



# for k in range(train.shape[0]):

    

#     # FIND OVERLAP

#     text1 = " "+" ".join(train.loc[k,'text'].split())

#     text2 = " ".join(train.loc[k,'new_selected_text'].split())

#     idx = text1.find(text2)

#     chars = np.zeros((len(text1)))

#     chars[idx:idx+len(text2)]=1

#     if text1[idx-1]==' ': chars[idx-1] = 1 

#     enc = tokenizer.encode(text1) 

        

#     # ID_OFFSETS

#     offsets = []; idx=0

#     for t in enc.ids:

#         w = tokenizer.decode([t])

#         offsets.append((idx,idx+len(w)))

#         idx += len(w)

    

#     # START END TOKENS

#     toks = []

#     for i,(a,b) in enumerate(offsets):

#         sm = np.sum(chars[a:b])

#         if sm>0: toks.append(i) 

        

#     s_tok = sentiment_id[train.loc[k,'sentiment']]

#     input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

#     attention_mask[k,:len(enc.ids)+3] = 1

#     if len(toks)>0:

#         start_tokens[k,toks[0]+2] = 1

#         end_tokens[k,toks[-1]+2] = 1





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

    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask_t[k,:len(enc.ids)+3] = 1
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

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
n_splits = 5
preds_start1 = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end1 = np.zeros((input_ids_t.shape[0],MAX_LEN)) 

for i in range(5):

    K.clear_session()

    model, padded_model = build_model()

    print('load model...')

    weight_fn = '/kaggle/input/with-processing/v0-roberta-%i.h5'%(i)

    load_weights(model, weight_fn)



    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=1)

    preds_start1 += preds[0]/n_splits

    preds_end1 += preds[1]/n_splits
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start1[k,])

    b = np.argmax(preds_end1[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    all.append(st)

test['selected_text1'] = all

# test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)
def reverse_preprocessing(text):



    text = text.replace(". . . .", "....")

    text = text.replace(". . .", "...")

    text = text.replace(". .", "..")

    text = text.replace("! ! ! !", "!!!!")

    text = text.replace("! ! !", "!!!")

    text = text.replace("! !", "!!")

    text = text.replace("? ? ? ?", "????")

    text = text.replace("? ? ?", "???")

    text = text.replace("? ?", "??")



    return text





def find_text_idx(text, selected_text):



    text_len = len(text)



    for start_idx in range(text_len):

        if text[start_idx] == selected_text[0]:

            for end_idx in range(start_idx+1, text_len+1):

                contained_text = "".join(text[start_idx: end_idx].split())

                # print("contained_text:", contained_text, "selected_text:", selected_text)

                if contained_text == "".join(selected_text.split()):

                    return start_idx, end_idx



    return None, None





def calculate_spaces(text, selected_text):



    selected_text = " ".join(selected_text.split())

    start_idx, end_idx = find_text_idx(text, selected_text)

    # print("text:", text[start_idx: end_idx], "prediction:", selected_text)



    if start_idx is None:

        start_idx = 0

        print("----------------- error no start idx find ------------------")

        print("text:", text, "prediction:", selected_text)

        print("----------------- error no start idx find ------------------")



    if end_idx is None:

        end_idx = len(text)

        print("----------------- error no end idx find ------------------")

        print("text:", text, "prediction:", selected_text)

        print("----------------- error no end idx find ------------------")



    x = text[:start_idx]

    try:

        if x[-1] == " ":

            x = x[:-1]

    except:

        pass



    l1 = len(x)

    l2 = len(" ".join(x.split()))

    return l1 - l2, start_idx, end_idx





def pp_v2(text, predicted):



    text = str(text).lower()

    predicted = predicted.lower()

    predicted = predicted.strip()



    if len(predicted) == 0:

        return predicted



    predicted = reverse_preprocessing(str(predicted))



    spaces, index_start, index_end = calculate_spaces(text, predicted)



    if spaces == 1:

        if len(text[max(0, index_start-1): index_end+1]) <= 0 or text[max(0, index_start-1): index_end+1][-1] != ".":

            return text[max(0, index_start - 1): index_end]

        else:

            return text[max(0, index_start-1): index_end+1]

    elif spaces == 2:

        return text[max(0, index_start-2): index_end]

    elif spaces == 3:

        return text[max(0, index_start-3): index_end-1]

    elif spaces == 4:

        return text[max(0, index_start-4): index_end-2]

    else:

        return predicted
test["selected_text"] = test.apply(lambda x: pp_v2(x.text, x.selected_text1), axis=1)

test.sample(10)
# test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index = False)
