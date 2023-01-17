# imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm, trange
import string
import re
import sys
import math
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout, BatchNormalization, Activation
import random
from keras.preprocessing.sequence import pad_sequences
from time import time
from sklearn import metrics
from keras.optimizers import Adam
from keras.regularizers import l1_l2
# loading training data
df=pd.read_csv('../input/train.csv')
df=df[['question_text','target']]
df=df.dropna() # just in case
# now lets look at training data more closely, we need to see how skewed the training data is
pos_pct=df['target'].value_counts()[1]/len(df)
print(f'Caustic questions percentage is {pos_pct*100:.2f}%')
trans_table={key:' ' for key in string.punctuation} # punctuation removal table
def get_words(s):
    '''Function for counting words in a question'''
    s=s.translate(str.maketrans(trans_table)).lower().strip() # remove punctuation
    s=re.sub(' +',' ',s) # get rid of multiple spaces inside
    s=s.split(' ')
    return len(s)
# what is the average number of words in a question?
df['n_words']=df['question_text'].apply(get_words)
# distribution for good and taxic questions
max_word_len=60
df[(df['target']==0) & (df['n_words']<=max_word_len)]['n_words'].value_counts().sort_index().plot(
    label='Sincere',legend=True, color=(0,0.3,0))
ax=df[(df['target']==1) & (df['n_words']<=max_word_len)]['n_words'].value_counts().sort_index().plot(
    figsize=(7,3.5), logy=True,xlim=(0,max_word_len+1), label='Toxic',legend=True,
    color='r', title='Questions distribution',xticks=np.arange(0,max_word_len+1,5)).set_xlabel('Words')
# first, we dump all questions longer than 60 words
df=df[df['n_words']<=60]
## comment this block before committing unfinished model! Saves up to 4 minutes of time
# load embeddings and create embedding vocabulary

embedding_path='glove.840B.300d/glove.840B.300d.txt' # <embedding folder>/<embedding file name>
vocab={}
with open('../input/embeddings/' + embedding_path,'r') as f:
    for line_number,line in enumerate(tqdm(f)):
        key,values=line.split(' ')[0],line.split(' ')[1:]
        if not any(char in string.punctuation for char in key): 
            # we do this filter because  we will remove punctuation anyway
            vocab[key]=np.asarray(values,dtype='float32')
print(f'Total of {len(vocab.keys())} words in vocabulary')
# string preprocessing function, where we get rid of punctuation, 
# split string to a list of words
# after that we encode words to 300-D vectors
trans_table={key:' ' for key in string.punctuation} # punctuation removal table
unknown_word=np.zeros(300) # token for unknown word

def str_prep(s):
    s=s.translate(str.maketrans(trans_table)).lower().strip() # remove punctuation
    s=re.sub(' +',' ',s) # get rid of multiple spaces inside
    s=s.split(' ')
    ar=np.asarray([vocab.get(x,unknown_word) for x in s],dtype='float32')
    return ar
# simpler batch generator, padding to 60 words
def batch_gen(df,min_batch_size=1024,transformation_func=str_prep, training_mode=True):
    n_batches=math.ceil(len(df)/min_batch_size)
    while True:
        df=df.sample(frac=1).reset_index(drop=True)
        for batch in range(n_batches):
            start=batch*min_batch_size
            end=start+min_batch_size
            if batch==n_batches-1:
                X=df['question_text'][start:].apply(transformation_func)
                X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
                y=np.array(df['target'][start:])
            else:
                X=df['question_text'][start:end].apply(transformation_func)
                X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
                y=np.array(df['target'][start:end])
            yield X, y
# train/validation split function

def train_val_split(df,frac=0.2):
    df=df.sample(frac=1).reset_index(drop=True) # random shuffling

    df_sincere=df[df['target']==0]
    df_taxic=df[df['target']==1]
    
    sincere_border=int(len(df_sincere)*(1-frac))
    taxic_border=int(len(df_taxic)*(1-frac))
    
    df_train=pd.concat([df_sincere[:sincere_border],df_taxic[:taxic_border]])
    df_val=pd.concat([df_sincere[sincere_border:],df_taxic[taxic_border:]])
    
    print('Training data peppered (prepared)!')
    return df_train.sample(frac=1).reset_index(drop=True), df_val.sample(frac=1).reset_index(drop=True)
# train model on data
def train_model(train_df,val_df,n_epochs=8,batch_size=1024):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                             input_shape=(60, 300)))
    model.add(Bidirectional(CuDNNLSTM(32)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(batch_gen(train_df, min_batch_size=batch_size), epochs=n_epochs,
                        steps_per_epoch=math.ceil(len(train_df)/batch_size),
                        validation_data=batch_gen(val_df, min_batch_size=batch_size),
                        validation_steps=math.ceil(len(val_df)/batch_size),
                        verbose=True)
    return model
# get predictions with trained model
def get_predictions(df, model, min_batch_size=1024,transformation_func=str_prep):
    predictions=np.ndarray(shape=(0,1))
    print(f'Total values to predict: {len(df)}')
    n_batches=math.ceil(len(df)/min_batch_size)
    for batch in trange(n_batches):
        start=batch*min_batch_size
        end=start+min_batch_size
        if batch==n_batches-1:
            X=df['question_text'][start:].apply(transformation_func)
            X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
            y_predicted=model.predict(X)
        else:
            X=df['question_text'][start:end].apply(transformation_func)
            X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
            y_predicted=model.predict(X)
        predictions=np.append(predictions,y_predicted,axis=0)
    return predictions
# selecting best threshold value for current model
def select_best_threshold(y_true, y_predicted):
    # code for finding threshold taken from
    # https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork
    # dont forget to upvote that kernel:
    thresholds = []
    for thresh in np.arange(0.0, 1, 0.001):
        thresh = np.round(thresh, 3)
        res=metrics.f1_score(y_true, (y_predicted>thresh).astype(int))
        thresholds.append([thresh, res])
    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    print("Best threshold: ", best_thresh)
    return best_thresh
stack_size=3
model_stack={}
for ind in range(stack_size):
    print(f'\n=== Model number {ind+1} ===\n')
    # creating training and validation sets, these will be different for each model in stack
    train_df, val_df=train_val_split(df, frac=0.1)
    # training current model
    model=train_model(train_df,val_df, n_epochs=5)
    # predicting on validation data
    y_predicted=get_predictions(val_df,model)
    y_true=val_df['target'].values
    # selecting threshold
    best_thresh=select_best_threshold(y_true, y_predicted)
    model_stack[ind]=(model, best_thresh)
    print(f'Model number {ind+1} finished!')
# predict on test set
test_set=pd.read_csv('../input/test.csv')
test_set.head()
predictions={}
for key, (model, threshold) in model_stack.items():
    predictions[key]=(get_predictions(test_set,model)>threshold).astype(int)
preds=np.stack([value for value in predictions.values()],axis=1).reshape(-1,stack_size)
preds=np.average(preds,axis=1)
preds=np.around(preds,decimals=0).astype(int)
test_set['prediction']=preds
to_submit=test_set[['qid','prediction']]

to_submit.to_csv("submission.csv", index=False)
print('Submissions saved to file!')
