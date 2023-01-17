import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

print('TF version',tf.__version__)
def read_train():

    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

    train['text']=train['text'].astype(str)

    train['selected_text']=train['selected_text'].astype(str)

    return train



def read_test():

    test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

    test['text']=test['text'].astype(str)

    return test



def read_submission():

    test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

    return test

    

train_df = read_train()

test_df = read_test()

submission_df = read_submission()
def jaccard_improve(str1, str2): 

    str1=str1.lower()

    str2=str2.lower()    

    index=str1.find(str2) 

    text1=str1[:index]

    #print(text1)

    text2=str1[index:].replace(str2,'')

    words1=text1.split()

    words2=text2.split()

    #print(words1[-3:])



    if len(words1)>len(words2):

        words1=words1[-3:]

        mod_text=" ".join(words1)+" "+ str2

    else:

        words2=words2[0:2]

        mod_text=str2+" "+" ".join(words2)

    return mod_text 
def jaccard(str1, str2): 

    a = set(str(str1).lower().split())  

    b = set(str(str2).lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
print(len(train_df))

#train_df1=train_df
train_df['selected_text_mod']=train_df['selected_text']

train_df['mod']=0
# train_df['mod']=0

# for index,row in train_df.iterrows():

#     #print(row['text'])

#     #print(row['selected_text'])

#     res1=jaccard(row['text'],row['selected_text_mod'])

#     res2=jaccard(row['text'],row['selected_text'])

    

#     if res1<0.5 and row['mod']==0:

#         mod_text=jaccard_improve(row['text'],row['selected_text'])

#         train_df.at[index,'mod']=1

#         train_df.at[index,'selected_text']=mod_text

# #         print('____________1')

# #         print(mod_text)

# #         print(row['text'])

# #         print(row['selected_text'])

# #         print('____________2')

#         res2=jaccard(row['text'],mod_text)

#     else:

#         train_df.at[index,'selected_text']=row['selected_text_mod']

    

#     train_df.at[index,'score1']=res1

#     train_df.at[index,'score2']=res2

    

#     #print(res1)

    

#     #print(res1)

#     #train_df.at[index,'score']=res1

 
# print(len(train_df[train_df.score1!=train_df.score2]))



# train_df[train_df.score1!=train_df.score2]



# #print(len(train_df[train_df.score>0.9]))

# train_df2=train_df[train_df.score>0.9]
#train_df = train_df2
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
ct = train_df.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
print(input_ids.shape)

print(attention_mask.shape)

print(token_type_ids.shape)

print(start_tokens.shape)

print(end_tokens.shape)
 



# for k in range(train_df.shape[0]):

    

#     # FIND OVERLAP

#     text1 = " "+" ".join(train_df.loc[k,'text'].split())

#     text2 = " ".join(train_df.loc[k,'selected_text'].split())

#     idx = text1.find(text2)

#     #text1='rx th as'

#     chars = np.zeros((len(text1)))

#     print("========1")

#     print(chars)

#     chars[idx:idx+len(text2)]=1

#     print(chars)

#     print(idx)

#     print(text1)

#     print(text2)

#     print(len(text2))

#     enc = tokenizer.encode(text1) 

#     print(enc)

#     print(text1)

#     if text1[idx-1]==' ': chars[idx-1] = 1 

#     print(chars)



#     offsets = []; idx=0    

#     for t in enc.ids:

#         w = tokenizer.decode([t])

#         #print(w)

#         #print(len(w))

#         offsets.append((idx,idx+len(w)))

#         idx += len(w)

#     #print(offsets)

        

        

#     #offsets.append((idx,idx+len(w)))

#     #idx += len(w)

        

#     # START END TOKENS

#     toks = []

#     for i,(a,b) in enumerate(offsets):

#         #print(a,b)

#         sm = np.sum(chars[a:b])

#         #print(chars[a:b])

#         #print(sm)

#         if sm>0: toks.append(i) 



#     s_tok = sentiment_id[train_df.loc[k,'sentiment']]

#     input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

#     attention_mask[k,:len(enc.ids)+5] = 1

#     if len(toks)>0:

#         start_tokens[k,toks[0]+1] = 1

#         end_tokens[k,toks[-1]+1] = 1            

            

#     print("========21")   

#     print(enc.ids)

#     print(enc)    

#     print(text1)

#     print(text2)

    

#     print(offsets)    

#     print(chars)

#     print(len(chars))

#     print(toks)

#     print(len(toks))



#     print(input_ids[k,:] )

#     print(s_tok)

#     print( start_tokens[k,])

#     print( end_tokens[k,])

#     print(attention_mask[k,:])

#     print(toks)

#     print([0] + enc.ids + [2,2] + [s_tok] + [2])

#     print("========2")

    

    
ct = train_df.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train_df.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train_df.loc[k,'text'].split())

    text2 = " ".join(train_df.loc[k,'selected_text'].split())

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

        

    s_tok = sentiment_id[train_df.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc.ids)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
ct = test_df.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test_df.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test_df.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test_df.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1
def scheduler(epoch):

    return 3e-5 * 0.2**epoch
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

    #x1 = tf.keras.layers.ReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    #x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)    

    model.compile(loss='binary_crossentropy', optimizer=optimizer)



    return model

n_splits = 5
# #input_ids

# #train_df.sentiment.values



# print(len(train_df))



# train_df1=train_df[:1000]

# print(len(train_df1))



# input_ids1=input_ids[:1000]





# skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777)

# for fold,(idxT,idxV) in enumerate(skf.split(input_ids1,train_df1.sentiment.values)):

#     print(idxV)

#     print(len(idxV))
# jac = []; VER='v6'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

# oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

# oof_end = np.zeros((input_ids.shape[0],MAX_LEN))





# skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=777)



# #for fold,(idxT,idxV) in enumerate(skf.split(input_ids1,train_df1.sentiment.values)):

# for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train_df.sentiment.values)):



#     print('#'*25)

#     print('### FOLD %i'%(fold+1))

#     print('#'*25)

    

#     K.clear_session()

#     model = build_model()

        

#     reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)



#     sv = tf.keras.callbacks.ModelCheckpoint(

#         '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, 

#         save_best_only=True,

#         save_weights_only=True, mode='auto', save_freq='epoch')

        

#     hist = model.fit([input_ids[idxT,], attention_mask[idxT,], 

#                       token_type_ids[idxT,]], [start_tokens[idxT,], 

#                                                end_tokens[idxT,]], 

#                         epochs=5, batch_size=8, verbose=DISPLAY, 

#                      callbacks=[sv, reduce_lr],

#         validation_data=([input_ids[idxV,],attention_mask[idxV,],

#                           token_type_ids[idxV,]], 

#         [start_tokens[idxV,], end_tokens[idxV,]]))

    

#     print('Loading model...')

#     model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    

#     print('Predicting OOF...')

#     oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

     

    

#     # DISPLAY FOLD JACCARD

#     all = []

#     for k in idxV:

#         a = np.argmax(oof_start[k,])

#         b = np.argmax(oof_end[k,])

#         if a>b: 

#             st = train_df.loc[k,'text'] # IMPROVE CV/LB with better choice here

#         else:

#             text1 = " "+" ".join(train_df.loc[k,'text'].split())

#             enc = tokenizer.encode(text1)

#             st = tokenizer.decode(enc.ids[a-1:b])

#         all.append(jaccard(st,train_df.loc[k,'selected_text']))

#     jac.append(np.mean(all))

#     print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

#     print(oof_start[idxV,])

#     print(oof_end[idxV,])

    

     

   

 



# # DISPLAY FOLD JACCARD

# all = []

# for k in idxV:

#     a = np.argmax(oof_start[k,])

#     b = np.argmax(oof_end[k,])

#     if a>b: 

#         st = train_df.loc[k,'text'] # IMPROVE CV/LB with better choice here

#     else:

#         text1 = " "+" ".join(train_df.loc[k,'text'].split())

#         enc = tokenizer.encode(text1)

#         st = tokenizer.decode(enc.ids[a-1:b])

#     all.append(jaccard(st,train_df.loc[k,'selected_text']))

# jac.append(np.mean(all))

# print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

# # print(oof_start[idxV,])

# # print(oof_end[idxV,])



 
#train_df.loc[10,'text']





#train_df.reset_index(inplace = True) 

#train_df
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

DISPLAY=1

for i in range(5):

    print('#'*25)

    print('### MODEL %i'%(i+1))

    print('#'*25)

    

    K.clear_session()

    model = build_model()

    #model.load_weights('../input/m6aprila/v6-roberta-%i.h5'%i)

    model.load_weights('../input/model8/v8-roberta-%i.h5'%i)



    #model.load_weights('v5-roberta-%i.h5'%i)



    print('Predicting Test...')

    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/n_splits

    preds_end += preds[1]/n_splits
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test_df.loc[k,'text']

    else:

        text1 = " "+" ".join(test_df.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])          

        st1=st.strip()

        if st1=='****' or  st1=='****!' or st1=='****!' or st1=='****!' or st1=='****,' or  st1=='****,' or  st1=='****.' or st1=='****.':

            #print(st1.strip())

            #print(text1)

            st=text1

        elif st1=='(good':   

            st='good'

        elif st1=='__joy':   

            st='joy'           

    all.append(st)
# import pandas as pd

# # train_df.to_csv('train_df.csv',index=False)



# test_df=pd.read_csv('../input/submission/submission_v2.csv')



# # for index,row in test_df.iterrows():

# #     row['selected_text']

# i=0

# for index,row in test_df.iterrows():

#     #print(row['selected_text'])

#     if len(row['selected_text'])>100:

#         #print(row['selected_text'])

#         test_df.at[index,'selected_text']=''

#         i=i+1

# print(i)
test_df['selected_text'] = all

test_df[['textID','selected_text']].to_csv('submission.csv',index=False)

 





# from distutils.dir_util import copy_tree

# todir='/kaggle/working'

# fromdirc='../input/tweet-sentiment-extraction'

# copy_tree(fromdirc,todir)



# import os

# os.chdir('/kaggle')

# os.getcwd()

# os.listdir()
# # import os

# # os.getcwd()



# import shutil

# source='/kaggle/test1.csv'

# destination='/kaggle/working/test1.csv'

# dest = shutil.copyfile(source, destination)

# import os

# import shutil

# for subdir, dirs, files in os.walk('/kaggle/working/'):

#     for file in files:

#         if '.h5' in file:

#           #print(file) #file

#           source='/kaggle/working'+str('/')+file

#           destination='../input/model4'+str('/')+file

#           destination='/kaggle'+str('/')+file

#           print(source)  

#           #print(destination)  

#           dest = shutil.copyfile(source, destination) 
# import os

# for subdir, dirs, files in os.walk('../input/model4/'):

#     for file in files:

#       print(file) #file
# from IPython.display import FileLink, FileLinks

# FileLinks('.') #lists all downloadable files on server