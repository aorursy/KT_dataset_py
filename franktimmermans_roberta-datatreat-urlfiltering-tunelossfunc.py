import numpy as np

import matplotlib.pyplot as plt

import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

import math

import timeit

import pickle

print('TF version',tf.__version__)
MAX_LEN = 100

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

EPOCHS = 7 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

SEED = 88888

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')

train.head()
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

test.head()
empties=[]

for x in range(0,train.shape[0]):

    text1=train['text'][x]

    if text1=="":

        empties.append(x)

        train=train.drop([x])

train=train.reset_index(drop=True)
empties2=[]

url_parts=['http://','https://','www.']

for x in range(0,train.shape[0]):

    text1=train['text'][x]

    for y in range(0,len(url_parts)):

        url_part=url_parts[y]

        idx = text1.find(url_part)

        if idx>=0:

            if idx<3:

                len_url=text1[idx:].find(' ')

                if len_url==-1:

                    empties2.append(x)

                    train=train.drop([x])

train=train.reset_index(drop=True)
def lower_text(text):

    return text.lower()

train['text_cleaned']=train['text'].apply(lambda x : lower_text(x))

train['selected_text_cleaned0']=train['selected_text'].apply(lambda x : lower_text(x))

test['text_cleaned']=test['text'].apply(lambda x : lower_text(x))
def Fix_Spaces(text):

    return " ".join(text.split())

train['text_cleaned']=train['text_cleaned'].apply(lambda x : Fix_Spaces(x))

train['selected_text_cleaned0']=train['selected_text_cleaned0'].apply(lambda x : Fix_Spaces(x))

test['text_cleaned']=test['text_cleaned'].apply(lambda x : Fix_Spaces(x))
test['selected_text']=-1

test.head()

df_data=pd.concat([train,test],ignore_index=True, sort=True)
df_data['text_cleaned_0']=df_data['text_cleaned']
tic = timeit.default_timer()



replace_text='[url]'

enc_url_replace=tokenizer.encode(replace_text).ids



MAX_LEN=200

Tokenizer_indices_orig = np.zeros((df_data.shape[0],MAX_LEN))

Tokenizer_indices_cleaned = np.zeros((df_data.shape[0],MAX_LEN))

Tokenizer_encoding_orig = np.zeros((df_data.shape[0],MAX_LEN))

Tokenizer_encoding_cleaned = np.zeros((df_data.shape[0],MAX_LEN))



url_parts=['http://','https://',' www.']

for x in range(0,df_data.shape[0]):

    text1=df_data['text_cleaned'][x]

    

    enc1 = tokenizer.encode(text1)

    Tokenizer_encoding_1=np.zeros((MAX_LEN))

    Tokenizer_encoding_1[0:len(np.array(enc1.ids))]=np.array(enc1.ids)

    Tokenizer_indices_1=np.zeros((MAX_LEN))

    Tokenizer_indices_1[0:len(np.array(enc1.ids))]=np.array((list(range(0, len(enc1.ids)))))+1

    

    Tokenizer_encoding_orig[x,0:len(np.array(enc1.ids))]=np.array(enc1.ids)

    Tokenizer_indices_orig[x,0:len(np.array(enc1.ids))]=np.array((list(range(0, len(enc1.ids)))))+1

    

    for y in range(0,len(url_parts)):

        F=True

        while F:

            url_part=url_parts[y]

            text1=' '+text1

            idx = text1.find(url_part)

            

            if idx != -1:

                if y==2:

                    idx=idx+1

                

                len_url=text1[idx:].find(' ')

                if len_url==-1 or len(text1)==idx+len_url+1:

                    text2=text1[:idx]

                    if text1[idx-1]==' ':

                        text2=text1[:idx-1]

                    text2=text2+' '+replace_text

                else:

                    part1=text1[:idx]

                    part2=text1[idx+len_url:]

                    text2=part1+replace_text+part2

                

                text1=text1[1:]

                text2=text2[1:]

                enc1 = tokenizer.encode(text1)

                enc2 = tokenizer.encode(text2)

                

                Tokenizer_encoding_1=np.zeros((MAX_LEN))

                Tokenizer_encoding_2=np.zeros((MAX_LEN))

                Tokenizer_encoding_1[0:len(np.array(enc1.ids))]=np.array(enc1.ids)

                Tokenizer_encoding_2[0:len(np.array(enc2.ids))]=np.array(enc2.ids)

                

                a=0

                part1_tokenized_indices=[]

                while Tokenizer_encoding_2[a]==Tokenizer_encoding_1[a]:

                    part1_tokenized_indices.append(Tokenizer_indices_1[a])

                    a=a+1

                part1_tokenized_indices=np.array(part1_tokenized_indices)

                

                a1=np.where(Tokenizer_encoding_1 == 0)[0].item(0)

                a2=np.where(Tokenizer_encoding_2 == 0)[0].item(0)

                part2_tokenized_indices=[]

                while Tokenizer_encoding_2[a2]==Tokenizer_encoding_1[a1]:

                    part2_tokenized_indices.append(Tokenizer_indices_1[a1])

                    a1=a1-1

                    a2=a2-1

                part2_tokenized_indices=np.flip(np.array(part2_tokenized_indices))

                

                part3_length=a2-a+1

                Available_space=a1-a+1

                

                if part3_length<0:

                    part2_tokenized_indices=part2_tokenized_indices[abs(part3_length):]     # Happens once for index 28430, the end of part1 is the same as the start of part2. a comma is counted double.

                

                if len(part1_tokenized_indices)!=0:

                    part3_tokenized_indices=np.array([part2_tokenized_indices[0]-1 , part2_tokenized_indices[0]-1 , part2_tokenized_indices[0]-1])

                else:

                    part3_tokenized_indices=np.array([1 , 1 , 1])

                

                new_indices=np.append(part1_tokenized_indices , part3_tokenized_indices)

                new_indices=np.append(new_indices , part2_tokenized_indices)

                Tokenizer_indices_2=new_indices

                

                text1=text2

                Tokenizer_encoding_1=Tokenizer_encoding_2

                Tokenizer_indices_1=Tokenizer_indices_2

            else:

                F=False

                text1=text1[1:]

        

    Tokenizer_indices_cleaned[x,0:len(Tokenizer_indices_1)]=Tokenizer_indices_1

    Tokenizer_encoding_cleaned[x,0:len(Tokenizer_encoding_1)]=Tokenizer_encoding_1

    df_data['text_cleaned'][x]=text1



toc = timeit.default_timer()

print(toc-tic)
difference_URL_library=[]

difference_URL_locs=[]

for x in range(0,df_data.shape[0]):

    if df_data['text_cleaned'][x]!=df_data['text_cleaned_0'][x]:

        difference_URL_locs.append([x])

        difference_URL_library.append(df_data['text_cleaned_0'][x])

len(difference_URL_library)

length1=[]

index=[]

for x in range(0,df_data.shape[0]):

    text=df_data['text_cleaned'][x]

    enc1 = tokenizer.encode(text)

    length1.append(len(enc1.ids))

    if length1[x] >= max(length1):

        index.append(x)



MAX_LEN=max(length1)

MAX_LEN=MAX_LEN+5

Tokenizer_encoding_cleaned_MaxLen=Tokenizer_encoding_cleaned[:,:MAX_LEN]

Tokenizer_indices_cleaned_MaxLen=Tokenizer_indices_cleaned[:,:MAX_LEN]

del train, test

train=df_data[df_data.selected_text!=-1]

train.shape
test=df_data[df_data.selected_text==-1]

test.drop('selected_text',axis=1,inplace=True)

test=test.reset_index(drop=True)

test.shape
train_mislabeled2 = train.groupby(['text_cleaned']).nunique().sort_values(by='sentiment', ascending=False)

train_mislabeled2 = train_mislabeled2[train_mislabeled2 ['sentiment'] > 1]['sentiment']

train_mislabeled2.index.tolist()

print(train_mislabeled2)



train_mislabeled3 = train.groupby(['text_cleaned']).nunique().sort_values(by='selected_text_cleaned0', ascending=False)

train_mislabeled3 = train_mislabeled3[train_mislabeled3 ['selected_text_cleaned0'] > 1]['selected_text_cleaned0']

train_mislabeled3.index.tolist()

print(train_mislabeled3)



# train.index[train['text_cleaned'] == "lost luggage? sorry to hear. you should check out our selection of travel luggage here: http://budurl.com/9mua"].tolist()



# print(train['text'][9727])

# print(train['text_cleaned'][9727])

# print(train['selected_text_cleaned0'][9727])

# print(train['selected_text'][9727])

# print(train['sentiment'][9727])

# 

# print([])

# print(train['text'][13528])

# print(train['text_cleaned'][13528])

# print(train['selected_text_cleaned0'][13528])

# print(train['selected_text'][13528])

# print(train['sentiment'][13528])
train['sentiment'][16438]='positive'

train['selected_text_cleaned0'][11431]="holy **** it`s super sunny, friday and whitsun, my tube is deeeesearted. wish i was in the park"

train['selected_text'][11431]="Holy **** it's super sunny, Friday and Whitsun, my tube is deeeesearted. Wish I was in the park"

train['sentiment'][11431]='neutral'

train['selected_text_cleaned0'][13897]="holy **** it`s super sunny, friday and whitsun, my tube is deeeesearted. wish i was in the park"

train['selected_text'][13897]="Holy **** it`s super sunny, Friday and Whitsun, my tube is deeeesearted. Wish I was in the park"

train['selected_text_cleaned0'][95]='happy'

train['selected_text'][95]='Happy'

train['sentiment'][8291]='positive'

train['sentiment'][21234]='positive'

train['selected_text_cleaned0'][5303]="lol"

train['selected_text'][5303]="lol"

train['sentiment'][5303]='positive'

train['sentiment'][4830]='positive'

train['sentiment'][22933]='positive'

train['selected_text_cleaned0'][13528]="sorry"

train['selected_text'][13528]="Sorry"

train['sentiment'][13528]='negative'

train['selected_text_cleaned0'][6403]="well us brits have to wait a few more days for it! i thought it was all gonna released at once! i guess it`s worth the wait!"

train['selected_text'][6403]="well us Brits have to wait a few more days for it!  I thought it was all gonna released at once! I guess it`s worth the wait!"

train['sentiment'][6403]='neutral'

train['selected_text_cleaned0'][14964]="happy"

train['selected_text'][14964]="Happy"

train['selected_text_cleaned0'][18573]="happy"

train['selected_text'][18573]="happy"

train['selected_text_cleaned0'][14199]="sad"

train['selected_text'][14199]="sad"

train['selected_text_cleaned0'][14947]="sad"

train['selected_text'][14947]="sad"

train['selected_text_cleaned0'][9305]="happy"

train['selected_text'][9305]="HAPPY"

train['selected_text_cleaned0'][15817]="happy"

train['selected_text'][15817]="Happy"

train['selected_text_cleaned0'][10620]="instant internet marketing empire! + *bonus* recoup your investment in 24 hours or less"

train['selected_text'][10620]="Instant Internet Marketing EMPIRE! + *BONUS* recoup your investment in 24 hours or less"

train['selected_text_cleaned0'][20251]="happy"

train['selected_text'][20251]="happy"

train['selected_text_cleaned0'][5547]="happy"

train['selected_text'][5547]="happy"

train['selected_text_cleaned0'][11961]="thank you. we had a blast"

train['selected_text'][11961]="thank you.  we had a blast"

train['selected_text_cleaned0'][24046]="joy!!"

train['selected_text'][24046]="joy!!"

train['selected_text_cleaned0'][5105]="happy"

train['selected_text'][5105]="happy"

train['selected_text_cleaned0'][3029]="happy"

train['selected_text'][3029]="Happy"

train['selected_text_cleaned0'][22683]="thanks!"

train['selected_text'][22683]="thanks!"

train['selected_text_cleaned0'][3259]="happy"

train['selected_text'][3259]="Happy"

train['selected_text_cleaned0'][15265]="happy"

train['selected_text'][15265]="happy"

train['selected_text_cleaned0'][19000]="happy"

train['selected_text'][19000]="HAPPY"

train['selected_text_cleaned0'][7121]="happy"

train['selected_text'][7121]="happy"

train['selected_text_cleaned0'][8487]="happy"

train['selected_text'][8487]="Happy"

train['selected_text_cleaned0'][13594]="happy"

train['selected_text'][13594]="Happy"

train['selected_text_cleaned0'][14355]="happy"

train['selected_text'][14355]="Happy"

train['selected_text_cleaned0'][1787]="good morning"

train['selected_text'][1787]="Good morning"

train['selected_text_cleaned0'][16641]="goodnight!"

train['selected_text'][16641]="goodnight!"

train['selected_text_cleaned0'][19045]="i cant afford"

train['selected_text'][19045]="i cant afford"

train['selected_text_cleaned0'][16767]="happy mothers day!!!"

train['selected_text'][16767]="Happy Mothers Day!!!"

train['selected_text_cleaned0'][8369]="good morning!"

train['selected_text'][8369]="Good Morning!"

train['selected_text_cleaned0'][10938]="nice one"

train['selected_text'][10938]="nice one"

train['selected_text_cleaned0'][24068]="g`night!"

train['selected_text'][24068]="G`night!"

train['selected_text_cleaned0'][7086]="thank you"

train['selected_text'][7086]="Thank you"

train['selected_text_cleaned0'][21612]="awesome to meet you all lol,"

train['selected_text'][21612]="awesome to meet you all lol,"

train['selected_text_cleaned0'][19490]="but **** it.."

train['selected_text'][19490]="but **** it.."

train['selected_text_cleaned0'][7546]="amazing"

train['selected_text'][7546]="amazing"
train['selected_text_cleaned0'][176]="can`t wait to see her bad n grown ****! lol"

train['selected_text'][176]="can`t wait to see her bad n grown ****! Lol"

train['sentiment'][176]='positive'

train['selected_text_cleaned0'][254]="lol :p"

train['selected_text'][254]="lol :p"

train['sentiment'][254]='positive'

train['selected_text_cleaned0'][851]="lol, i`ve done that one b4 i`m a victim 2 that! lol"

train['selected_text'][851]="lol, i`ve done that one b4  i`m a victim 2 that! lol"

train['sentiment'][851]='positive'

train['selected_text_cleaned0'][1234]="lol missed you ha bye hun ****"

train['selected_text'][1234]="lol  missed you ha bye hun ****"

train['sentiment'][1234]='positive'

train['selected_text_cleaned0'][1570]="because i had a blast!!"

train['selected_text'][1570]="because i had a blast!!"

train['sentiment'][1570]='positive'

train['selected_text_cleaned0'][2124]="lol - that`s what hubby`s are there for"

train['selected_text'][2124]="lol - that`s what hubby`s are there for"

train['sentiment'][2124]='positive'

train['selected_text_cleaned0'][2854]="but a good time was had by all"

train['selected_text'][2854]="but a good time was had by all"

train['sentiment'][2854]='positive'

train['selected_text_cleaned0'][2908]="lol hi boys!"

train['selected_text'][2908]="LOL  Hi boys!"

train['sentiment'][2908]='positive'

train['selected_text_cleaned0'][3503]=";) xoxo"

train['selected_text'][3503]=";) xoxo"

train['sentiment'][3503]='positive'

train['selected_text_cleaned0'][3847]="totally worth it!! great movie cool 3d glasses!"

train['selected_text'][3847]="totally worth it!! great movie cool 3D glasses!"

train['sentiment'][3847]='positive'

train['selected_text_cleaned0'][9895]="how much fun kyle is!"

train['selected_text'][9895]="how much fun kyle is!"

train['sentiment'][9895]='positive'

train['selected_text_cleaned0'][10347]="i miss you too!! and don`t say 'damn'!!! lol"

train['selected_text'][10347]="I miss you too!!   And don`t say 'damn'!!!  lol"

train['sentiment'][10347]='positive'

train['selected_text_cleaned0'][10827]="lol only if you make me that cookie"

train['selected_text'][10827]="Lol Only if you make me that cookie"

train['sentiment'][10827]='positive'

train['selected_text_cleaned0'][11423]="lol xoxo"

train['selected_text'][11423]="lol xoxo"

train['sentiment'][11423]='positive'

train['selected_text_cleaned0'][15558]="lol i feel like drunk right now..."

train['selected_text'][15558]="LOL I feel like drunk right now"

train['sentiment'][15558]='positive'

train['selected_text_cleaned0'][18980]="no u miss me !!!!! lol"

train['selected_text'][18980]="No u miss me !!!!! LOL"

train['sentiment'][18980]='positive'

train['selected_text_cleaned0'][19647]="lol"

train['selected_text'][19647]="lol"

train['sentiment'][19647]='positive'

train['selected_text_cleaned0'][20706]="yen lol but i can only get the vid on my phone and ipod cant find the song lol"

train['selected_text'][20706]="yen lol but i can only get the vid on my phone and ipod cant find the song  lol"

train['sentiment'][20706]='positive'

train['sentiment'][26899]='positive'

train['selected_text_cleaned0'][27468]="lol i know and haha"

train['selected_text'][27468]="lol i know  and haha"

train['sentiment'][27468]='positive'

train['selected_text_cleaned0'][17539]="happy mothers day"

train['selected_text'][17539]="HAPPY MOTHERS DAY"

train['sentiment'][17539]='positive'

train['selected_text_cleaned0'][24850]="good work"

train['selected_text'][24850]="Good work"

train['sentiment'][24850]='positive'
train['selected_text_cleaned0'][870]="oh no!!"

train['selected_text'][870]="Oh no!!"

train['sentiment'][870]='negative'

train['sentiment'][982]='negative'

train['sentiment'][1871]='negative'

train['sentiment'][2665]='negative'

train['sentiment'][2976]='negative'

train['selected_text_cleaned0'][7457]="sorry your still not well"

train['selected_text'][7457]="sorry your still not well"

train['sentiment'][7457]='negative'

train['selected_text_cleaned0'][9475]="sorry for your loss dear"

train['selected_text'][9475]="sorry for your loss dear"

train['sentiment'][9475]='negative'

train['sentiment'][11891]='negative'

train['sentiment'][14558]='negative'

train['sentiment'][21604]='negative'

train['sentiment'][26664]='negative'
# In[ ]:

ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



start_tokens_values=[]

end_tokens_values=[]

start_tokens_values_cleaned=[]

end_tokens_values_cleaned=[]

start_tokens_cleaned=np.zeros((train.shape[0],MAX_LEN),dtype='int32')

end_tokens_cleaned=np.zeros((train.shape[0],MAX_LEN),dtype='int32')



error_list=[]

for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text_cleaned_0'].split())

    text2 = " ".join(train.loc[k,'selected_text_cleaned0'].split())

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

    

    start_tokens[k,toks[0]+1] = 1

    end_tokens[k,toks[-1]+1] = 1

    start_tokens_values.append(toks[0]+1)

    end_tokens_values.append(toks[-1]+1)

    

    # Below convert from original start and end toks to new start and end toks

    Tokenizer_indices=Tokenizer_indices_cleaned_MaxLen[k,:]

    start_token_2=[]

    for x in range(MAX_LEN):

        if Tokenizer_indices[x]==start_tokens_values[k]:

            start_token_2.append(x+1)

    if len(start_token_2)>1:

        start_token_2=list([start_token_2[0]])  # If more than 1 index found, pick the first one.

    if len(start_token_2)==0:

        F=True

        x=0

        while F:

            if Tokenizer_indices[x]<=start_tokens_values[k]:

                x=x+1

            else:

                F=False

        if x==0:

            x=1

        start_token_2=list([x])

    

    end_token_2=[]

    for x in range(MAX_LEN):

        if Tokenizer_indices[x]==end_tokens_values[k]:

            end_token_2.append(x+1)

    if len(end_token_2)>1:

        end_token_2=list([end_token_2[-1]])  # If more than 1 index found, pick the last one.

    if len(end_token_2)==0:

        for x in range(MAX_LEN):

            if Tokenizer_indices[x]>=end_tokens_values[k]:

                end_token_2.append(x+1)

        if len(end_token_2)>=1:

            end_token_2=list([end_token_2[0]])

        else:

            F=True

            x=0

            while F:

                if Tokenizer_indices[x]!=0:

                    x=x+1

                else:

                    F=False

            end_token_2=list([x])

    if end_token_2<start_token_2:

        start_token_2=1

        end_token_2=end_tokens_values[k]

        error_list.append(k)

    

    start_tokens_values_cleaned.append(np.int(start_token_2[0]))

    end_tokens_values_cleaned.append(np.int(end_token_2[0]))

    start_tokens_cleaned[k,np.int(start_token_2[0])] = 1

    end_tokens_cleaned[k,np.int(end_token_2[0])] = 1

    

    text1_cleaned = " "+train.loc[k,'text_cleaned']

    enc_cleaned = tokenizer.encode(text1_cleaned) 

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc_cleaned.ids)+5] = [0] + enc_cleaned.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc_cleaned.ids)+5] = 1
ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

    # INPUT_IDS

    text1_cleaned = " "+" ".join(test.loc[k,'text_cleaned'].split())

    enc_cleaned = tokenizer.encode(text1_cleaned)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc_cleaned.ids)+5] = [0] + enc_cleaned.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc_cleaned.ids)+5] = 1

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

    weight_ratio=0.05

    weight_offset=1

    

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    

    true_index=tf.argmax(y_true, axis=1)

    pred_index=tf.argmax(y_pred, axis=1)

    true_index=tf.cast(true_index, tf.float32)

    pred_index=tf.cast(pred_index, tf.float32)

    

    weight=abs(true_index-pred_index)*weight_ratio+weight_offset

    weight=tf.cast(weight, tf.float32)

    

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = loss*weight

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

    

    x1 = tf.keras.layers.Dropout(0.2)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.2)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

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
# In[ ]:

def scheduler(epoch):

    return 3e-5 * 0.5**epoch
# In[ ]:

jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

val_epoch_start = np.zeros((input_ids.shape[0],MAX_LEN))

val_epoch_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED) #originally 5 splits

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()

    

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens_cleaned[idxT,], end_tokens_cleaned[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens_cleaned[idxV,], end_tokens_cleaned[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    

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

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[reduce_lr],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        

        # Save model only when jaccard score on validation improves for the current epoch

        val_epoch_start[idxV,],val_epoch_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

        all = []

        for idxV_val in idxV:

            a = np.argmax(val_epoch_start[idxV_val,])

            b = np.argmax(val_epoch_end[idxV_val,])

            

            a2=Tokenizer_indices_cleaned_MaxLen[idxV_val,a-1]

            b2=Tokenizer_indices_cleaned_MaxLen[idxV_val,b-1]

            if a>b: 

                st = train.loc[idxV_val,'text']

            else:

                text1 = " "+" ".join(train.loc[idxV_val,'text'].split())

                enc = tokenizer.encode(text1)

                st = tokenizer.decode(enc.ids[int(a2-1):int(b2)])

            all.append(jaccard(st,train.loc[idxV_val,'selected_text']))

        score_value=np.mean(all)

        print('>>>> Jaccard =',score_value)

        

        if epoch==1:

            score_value0=score_value

            print('save weights')

            save_weights(model, weight_fn)

            print()

        else:

            if score_value>score_value0:

                score_value0=score_value

                print('save weights')

                save_weights(model, weight_fn)

            print()

    

    print('Loading model...')

    load_weights(model, weight_fn)

    

    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

            

        a2=Tokenizer_indices_cleaned_MaxLen[k,a-1]

        b2=Tokenizer_indices_cleaned_MaxLen[k,b-1]

        if a>b: 

            st = train.loc[k,'text']

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[int(a2-1):int(b2)])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()

    

    with open('Predictions_test_%i'%(fold)+'.pickle', 'wb') as f:

        pickle.dump([preds_start, preds_end], f)
# In[ ]:

with open('Predictions_Validation.pickle', 'wb') as f:

    pickle.dump([oof_start, oof_end], f)

print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
all = []

log_indices=[]

for k in range(oof_start.shape[0]):

    a = np.argmax(oof_start[k,])

    b = np.argmax(oof_end[k,])

    

    a2=Tokenizer_indices_cleaned_MaxLen[k,a-1]

    b2=Tokenizer_indices_cleaned_MaxLen[k,b-1]

    if a>b: 

        log_indices.append(k)

        st = train.loc[k,'text']

    else:

        text1 = " "+" ".join(train.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[int(a2-1):int(b2)])

    all.append(jaccard(st,train.loc[k,'selected_text']))

print('>>>> Jaccard =',np.mean(all))

print()
all = []

log_indices=[]

for k in range(oof_start.shape[0]):

    a = np.argmax(oof_start[k,])

    b = np.argmax(oof_end[k,])

    

    a2=Tokenizer_indices_cleaned_MaxLen[k,a-1]

    b2=Tokenizer_indices_cleaned_MaxLen[k,b-1]

    if train.loc[k,'sentiment']=='neutral':

        st = train.loc[k,'text']

    else:

        if a>b: 

            log_indices.append(k)

            st = train.loc[k,'text']

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[int(a2-1):int(b2)])

    all.append(jaccard(st,train.loc[k,'selected_text']))

print('>>>> Jaccard =',np.mean(all))

print()
all = []

offset_train=train.shape[0]

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    

    k_shift=k+offset_train

    a2=Tokenizer_indices_cleaned_MaxLen[k_shift,a-1]

    b2=Tokenizer_indices_cleaned_MaxLen[k_shift,b-1]

    if test.loc[k,'sentiment']=='neutral':

        st = test.loc[k,'text']

    else:

        if a>b: 

            st = test.loc[k,'text']

        else:

            text1 = " "+" ".join(test.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[int(a2-1):int(b2)])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)