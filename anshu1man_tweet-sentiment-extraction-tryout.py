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
#!pip install bert
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import tokenizers

import tensorflow as tf

import transformers

from keras.models import Model,Sequential

from keras.callbacks import ModelCheckpoint



MAX_LEN =100

PATH='/kaggle/input/tf-roberta/'

TRAIN_BATCH_SIZE= 32

VALID_BATCH_SIZE=16



TOKENIZER =  tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{PATH}/vocab-roberta-base.json", 

    merges_file=f"{PATH}/merges-roberta-base.txt", 

    lowercase=True,

    add_prefix_space=True

)
df= pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').dropna().reset_index(drop=True)

df.head()
sent_id = {

        'positive': 1313,

        'negative': 2430,

        'neutral': 7974

}
def preprocess(item):

        tweet=" ".join(str(item['text']).split())

        selected_text=" ".join(str(item['selected_text']).split())

        



        len_sel_text= len(selected_text)

        idx0 =None

        idx1 =None

        for ind in (i for i,e in enumerate(tweet) if e==selected_text[0]):

            if tweet[ind:ind+len_sel_text]==selected_text:

                idx0= ind

                idx1= ind+ len(selected_text)-1

                break

        

        char_targets=[0]*len(tweet)



        if idx0!=None and idx1!=None:

            for j in range(idx0,idx1+1):

                    char_targets[j]=1;

        

        tok_tweet =TOKENIZER.encode(tweet)

        tok_tweet_tokens=tok_tweet.tokens

        tok_tweet_ids=tok_tweet.ids

        tok_tweet_offsets=tok_tweet.offsets[1:-1]



        targets=[]

        for j,(offset1,offset2) in enumerate(tok_tweet_offsets):

            if sum(char_targets[offset1:offset2])>0:

                targets.append(j)



        tok_tweet_ids= [0]+tok_tweet_ids+[2]

        tok_tweet_ids=tok_tweet_ids+[2]+[sent_id[item['sentiment']]]+[2]

        

        targets_start=[0]*(MAX_LEN)

        targets_end=[0]*(MAX_LEN)

        

        if len(targets)>0:

            targets_start[targets[0]+1]=1

            targets_end[targets[-1]+1]=1

        

        mask=[1]*len(tok_tweet_ids)

        

        padding_len= MAX_LEN- len(tok_tweet_ids)

        ids= tok_tweet_ids+[1]*padding_len

        mask= mask+[0]*padding_len

        token_type=[0]*MAX_LEN

        return {

            'ids': np.array(ids,dtype=np.int32),

            'mask': np.array(mask,dtype=np.int32),

            'tok_ids':np.array(token_type,dtype=np.int32),

            'start':np.array(targets_start,dtype=np.int32),

            'end':np.array(targets_end,dtype=np.int32)

            

        }
tr_ids,tr_att,tr_tok,tr_st,tr_end=[],[],[],[],[]

for x in range(df.shape[0]):

    out= preprocess(df.iloc[x])

    tr_ids.append(out['ids'])

    tr_att.append(out['mask'])

    tr_tok.append(out['tok_ids'])

    tr_st.append(out['start'])

    tr_end.append(out['end'])

# tr_ids== tf.convert_to_tensor(tr_ids,dtype=tf.int32)

# tr_att= tf.convert_to_tensor(tr_att,dtype=tf.int32)

# tr_tok= tf.convert_to_tensor(tr_tok,dtype=tf.int32)

# tr_st= tf.convert_to_tensor(tr_st,dtype=tf.int32)

# tr_end=tf.convert_to_tensor(tr_end,dtype=tf.int32)
tr_ids=np.array(tr_ids,dtype=np.int32)

tr_att= np.array(tr_att,dtype=np.int32)

tr_tok= np.array(tr_tok,dtype=np.int32)

tr_st= np.array(tr_st,dtype=np.int32)

tr_end=np.array(tr_end,dtype=np.int32)
tr_st.shape
def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)



    config = transformers.RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = transformers.TFRobertaModel.from_pretrained(PATH+

            'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids,attention_mask=att,token_type_ids=tok)



    x1 = tf.keras.layers.Conv1D(1,1)(x[0])

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)



    x2 = tf.keras.layers.Conv1D(1,1)(x[0])

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])



    return model
model= build_model()

model.summary()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
test_df= pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df.head()
def preprocesstest(item):

        tweet=" ".join(str(item['text']).split())

        tok_tweet =TOKENIZER.encode(tweet)

        tok_tweet_tokens=tok_tweet.tokens

        tok_tweet_ids=tok_tweet.ids

        tok_tweet_offsets=tok_tweet.offsets[1:-1]



        tok_tweet_ids= [0]+tok_tweet_ids+[2]  

        

        tok_tweet_ids=tok_tweet_ids+[2]+[sent_id[item['sentiment']]]+[2]

        mask=[1]*len(tok_tweet_ids)

        

        padding_len= MAX_LEN- len(tok_tweet_ids)

        ids= tok_tweet_ids+[1]*padding_len

        mask= mask+[0]*padding_len

  

        token_type=[0]*MAX_LEN

        return {

            'ids': tf.convert_to_tensor(ids,dtype=tf.int32),

            'mask': tf.convert_to_tensor(mask,dtype=tf.int32),

            'tok_ids':tf.convert_to_tensor(token_type,dtype=tf.int32)

        }

    
fold=1

jacks=[]

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=101)

for fold,(idxT,idxV) in enumerate(skf.split(tr_ids,df.sentiment.values)):

    print('*'*10)

    print('Fold ',fold+1)

    fold+=1

    

    model.fit(x=[tr_ids[idxT],tr_att[idxT],tr_tok[idxT]],y=[tr_st[idxT],tr_end[idxT]],epochs=4,validation_data=([tr_ids[idxV],tr_att[idxV],tr_tok[idxV]],[tr_st[idxV],tr_end[idxV]]),shuffle=True,batch_size=64)

    

    scores=[]

    for inx in idxV:

        sent=df['text'][inx]

        ids=tr_ids[inx]

        att=tr_att[inx]

        tok=tr_tok[inx]

        ids= np.reshape(ids,(1,len(ids)))

        att= np.reshape(att,(1,len(att)))

        tok= np.reshape(tok,(1,len(tok)))

        start,end = model.predict([ids,att,tok])

        stidx= np.argmax(start)

        stend= np.argmax(end)

        if stidx>stend:

            t=stidx

            stidx=stend

            stend=t

        text1 = " ".join(sent.split())

        enc = TOKENIZER.encode(text1)

        st = TOKENIZER.decode(enc.ids[stidx-1:stend+1])

        scores.append(jaccard(sent,st))

    jacks.append(np.mean(scores))

    print(np.mean(scores))

    print('*'*10)



print(np.mean(jacks))

output=[]

indices=[]

for x in range(test_df.shape[0]):

    sent=test_df.iloc[x]['text']

    inp = preprocesstest(test_df.iloc[x])

    ids= (inp['ids'])

    att = (inp['mask'])

    tok = inp['tok_ids']

    ids= tf.reshape(ids,(1,len(ids)))

    att= tf.reshape(att,(1,len(att)))

    tok= tf.reshape(tok,(1,len(tok)))

    start,end= model.predict([ids,att,tok])

    stidx= np.argmax(start)

    stend= np.argmax(end)

    if stidx>stend:

        t=stidx

        stidx=stend

        stend=t

        

    text1 = " ".join(sent.split())

    enc = TOKENIZER.encode(text1)

    st = TOKENIZER.decode(enc.ids[stidx-1:stend+1])

    output.append(st)

    indices.append([stidx,stend])
sub= pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub.loc[:,'selected_text']=output
sub.drop(['text','sentiment'],axis=1,inplace=True)

sub.head()
sub.to_csv("submission.csv", index=False)
# for x in range(79,100):

#     print("*"*25)

#     print(test_df.iloc[x]['text'])

#     print(indices[x])

#     print(output[x])