import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm as tqdm

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import confusion_matrix

import json

from pandas.io.json import json_normalize

#from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, LSTM, Dense,concatenate,TimeDistributed,SimpleRNN,GRU

from keras.models import Model

from keras.callbacks import EarlyStopping 

from keras import regularizers

import keras.backend as K

import tensorflow as tf

from numba import jit

import random

import collections

import gc

import copy

import sys



dir='/kaggle/input/data-science-bowl-2019/'

#dir='./'

nbatch, nstep, nreset, maxlen, nfold = 1500, 70, 3, 300, 6

a,b,c=-1,-2,-4  # coefficients of the loss matrix; power of 2 

random.seed(39)
#quadratic_kappa

@jit

def quadratic_kappa(actuals, preds, sample_weight, N=4):

    #if len(sample_weight)==1:

    #    sample_weight=np.ones(len(actuals),dtype='int32')

    w, MO = np.zeros((N,N)), np.zeros((N,N))

    for n in range(len(actuals)):

        MO[actuals[n]][preds[n]]+=sample_weight[n]

    for i in range(len(w)): 

        for j in range(len(w)):

            w[i][j] = float(((i-j)**2))

    act_hist, pred_hist = np.zeros(N), np.zeros(N)

    for n in range(len(actuals)):

        act_hist[actuals[n]]+=sample_weight[n]

        pred_hist[preds[n]]+=sample_weight[n]   

    ME = np.outer(act_hist, pred_hist);

    ME = ME/ME.sum()

    MO = MO/MO.sum()

    

    num, den = 0, 0

    for i in range(len(w)):

        for j in range(len(w)):

            num+=w[i][j]*MO[i][j]

            den+=w[i][j]*ME[i][j]

    return (1 - (num/den))
#train = pd.read_csv(dir+"train.csv",nrows=3000000)

train = pd.read_csv(dir+"train.csv")

train_labels = pd.read_csv(dir+"train_labels.csv")

#specs = pd.read_csv("specs.csv")

sample_submission = pd.read_csv(dir+"sample_submission.csv")

del train['timestamp']

del train['type']

del train['world']
#train.shape

ids = list(train_labels['installation_id'].unique())

train=train[train['installation_id'].isin(ids)]

list_title = train['title'].unique()

list_event_code = train['event_code'].unique()

list_event_code.sort()

list_atitle = train_labels['title'].unique()
def transform_df(train,nsplit=1):

    split = np.array_split(np.arange(len(train)), nsplit)

    for i in tqdm(range(len(split))):

        tmp1 = train['event_data'].iloc[split[i]].apply(json.loads).values

        for j in range(len(tmp1)):

            tmp2={}

            #if 'description' in tmp[j].keys():

            #    tmp2['description'] = tmp[j]['description']

            if 'round' in tmp1[j].keys():

                tmp2['round'] = tmp1[j]['round']

            if 'level' in tmp1[j].keys():

                tmp2['level'] = tmp1[j]['level']

            if 'correct' in tmp1[j].keys():

                tmp2['correct'] = tmp1[j]['correct']

            if 'misses' in tmp1[j].keys():

                tmp2['misses'] = tmp1[j]['misses']

            tmp1[j]=tmp2

        if i ==0:

            tmp=tmp1[0:]

        else:

            tmp=np.concatenate([tmp,tmp1])



    tmp=json_normalize(tmp)

    tmp.index=train.index

    train = pd.merge(train, tmp,how='inner',left_index=True, right_index=True)

    del tmp



    train['event_code_label']=le_event_code.transform(train['event_code'])

    del train['event_code']

    train['title_label']=le_title.transform(train['title'])

    

    return train



le_title = LabelEncoder()

le_title.fit(list_title)

le_atitle = LabelEncoder()

le_atitle.fit(list_atitle)

le_event_code = LabelEncoder()

le_event_code.fit(list_event_code)

train=transform_df(train,nsplit=10)
def make_set(train,fortest=False):

    label_sessions = list(train_labels['game_session'].unique())

    X,A,Y,T,ID=[],[],[],[],[]

    for i, iid in tqdm(train.groupby('installation_id', sort=False)):

        Xa=np.zeros((iid['game_session'].nunique(),len(list_title)+len(list_event_code)+8), dtype=np.float32)

        total_n, total_t, total_r, total_l, total_s, total_f, total_m = 0.00001, 0, 0, 0, 0, 0, 0 

        for n,(j, session) in enumerate(iid.groupby('game_session', sort=False)):

            tmp=session['event_code_label'].value_counts(sort=False)

            Xa[n,session['title_label'].iloc[0]]=1

            Xa[n,tmp.index+len(list_title)]=tmp/10

            Xa[n,-1]=len(session)/100

            Xa[n,-2]=session['game_time'].iloc[-1]/100000

            Xa[n,-3]=sum(session[session['round'] != np.nan]['round'])/100

            if np.isnan(Xa[n,-3]):

                Xa[n,-3]=0

            Xa[n,-4]=sum(session[session['level'] != np.nan]['level'])/100

            if np.isnan(Xa[n,-4]):

                Xa[n,-4]=0

            Xa[n,-5]=max(session[session['level'] != np.nan]['level'])/10

            if np.isnan(Xa[n,-5]):

                Xa[n,-5]=0

            Xa[n,-6]=sum(session['correct']==True)

            Xa[n,-7]=sum(session['correct']==False)

            Xa[n,-8]=sum(session['misses']==True)/10

            #print(Xa[n,-8:])

            if (fortest==False) and (j in label_sessions):

                Atmp=[le_atitle.transform([session['title'].values[0]])[0],total_n/100,total_t/total_n

                      , total_r/total_n, total_l/total_n, total_s/total_n, total_f/total_n, total_m/total_n]

                A.append(Atmp)

                X.append(Xa[0:n].copy())

                T.append(n)

                ID.append(i)

                Y.append(train_labels[train_labels['game_session']==j]['accuracy_group'].values[0])

                label_sessions.remove(j)

            if (fortest) and (n == iid['game_session'].nunique()-1):

                Atmp=[le_atitle.transform([session['title'].values[0]])[0],total_n/100,total_t/total_n

                      , total_r/total_n, total_l/total_n, total_s/total_n, total_f/total_n, total_m/total_n]

                A.append(Atmp)

                X.append(Xa[0:n].copy())

                T.append(n)

                ID.append(i)

            #total_n += 1

            #total_t += Xa[n,-2]

            #total_r += Xa[n,-3]

            #total_l += Xa[n,-4]

            #total_s += Xa[n,-5]

            #total_f += Xa[n,-6]

            #total_m += Xa[n,-7]

    return X,A,Y,T,ID



X,A,Y,T,ID=make_set(train)

del train

#gc.collect()

#print(len(X),len(A),len(Y),len(T))
print(len(X),len(A[0]),len(Y),len(T),len(ID))
#for padding of the time-series

def padding(X,A,Y=[],maxlen=maxlen):

    X_train = pad_sequences(X, maxlen=maxlen, dtype='float32', padding='pre', truncating='post', value=0.0)

    A_train = np.zeros((len(X),5), dtype=np.float32)

    #A_train = np.zeros((len(X),len(A[0])+4), dtype=np.float32)

    for i in range(len(X)):

        A_train[i,A[i][0]] = 1

        #A_train[i,5:] = A[i][1:]

    if len(Y)>0:

        Y_train = np.zeros((len(Y),4), dtype=np.int32)

        for i in range(len(Y)):

            Y_train[i,Y[i]] = 1

        return X_train,A_train,Y_train

    else:

        return X_train,A_train        



X_train,A_train,Y_train = padding(X,A,Y)

del X

gc.collect()

print(X_train.shape,A_train.shape,Y_train.shape)
def make_network():

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    atitle = Input(shape=(5,), name='atitle')

    x=inputs

    x = TimeDistributed(Dense(36,activation='tanh',kernel_regularizer=regularizers.l1(0.00001)))(x)

    x = TimeDistributed(Dense(18,activation='tanh',kernel_regularizer=regularizers.l1(0.0000)))(x)

    x = TimeDistributed(Dense(10,activation='tanh'))(x)

    #x = TimeDistributed(Dense(10,activation='relu'))(x)

    x = GRU(10,return_sequences=False)(x)

    y = concatenate([x, atitle])

    y = Dense(15,activation='tanh')(y)

    y = Dense(10,activation='tanh')(y)

    y = Dense(10,activation='tanh')(y)

    #y = Dense(10,activation='relu')(y)

    y = Dense(4,activation='softmax')(y)

    model = Model(inputs=[inputs,atitle], outputs=y)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    #model.summary()

    return model
le_id = LabelEncoder()

le_id.fit(ID)

nID=le_id.transform(ID)

nlabel=np.empty(len(ID))

for i in range(len(ID)):

    nlabel[i]=sum(e==ID[i] for e in ID)

id_sample_all=np.array(random.sample(range(len(set(ID))),len(set(ID))))

W_train=1/nlabel



lossmat=np.array([[0,a,b,c],[a,0,a,b],[b,a,0,a],[c,b,a,0]])

models = []

for ii in range(nfold):

    i=ii

    log, logs = np.zeros((nstep,8)), []    

    id_sample = np.array_split(id_sample_all, nfold)[i]

    test_sample = [e for e in range(len(ID)) if nID[e] in id_sample]

    train_sample = [e for e in range(len(Y_train)) if e not in test_sample]

    X_train1, A_train1, Y_train1, W_train1 = X_train[train_sample],A_train[train_sample],Y_train[train_sample], W_train[train_sample]

    X_train2, A_train2, Y_train2, W_train2 = X_train[test_sample],A_train[test_sample],Y_train[test_sample], W_train[test_sample]

    print(i,len(id_sample),len(set(test_sample)),len(set(train_sample)))

    maxacc=-1

    log2 = log.copy()

    for j in range(nreset):

        model = make_network()

        for s in tqdm(range(nstep)):

            history = model.fit([X_train1,A_train1], Y_train1, sample_weight=W_train1, batch_size=nbatch, 

                                epochs=1,shuffle=True,validation_data=([X_train2,A_train2],Y_train2),verbose=0)

            Yp=(np.matmul(model.predict([X_train1,A_train1],batch_size=2000),lossmat)).argmax(axis=1).astype(np.int32)

            log2[s,4] = quadratic_kappa(Y_train1.argmax(axis=1).astype(np.int32), Yp, sample_weight=W_train1)  

            log2[s,6] = quadratic_kappa(Y_train1.argmax(axis=1).astype(np.int32), Yp, sample_weight=np.ones(len(Y_train1)))  

            Yp=(np.matmul(model.predict([X_train2,A_train2],batch_size=2000),lossmat)).argmax(axis=1).astype(np.int32)

            log2[s,5] = quadratic_kappa(Y_train2.argmax(axis=1).astype(np.int32), Yp, sample_weight=W_train2)  

            log2[s,7] = quadratic_kappa(Y_train2.argmax(axis=1).astype(np.int32), Yp, sample_weight=np.ones(len(Y_train2)))  

            log2[s,0], log2[s,1]= history.history['loss'][0], history.history['val_loss'][0]

            log2[s,2], log2[s,3]= history.history['categorical_accuracy'][0], history.history['val_categorical_accuracy'][0]

            #print(s,log2[s])

            if -log2[s,1]+log2[s,3]+log2[s,4]+3*log2[s,5]>maxacc and s>20:

                model0=copy.deepcopy(model)

                maxacc=-log2[s,1]+log2[s,3]+log2[s,4]+3*log2[s,5]

                #print(maxacc)

        if j==0:

            log=log2.copy()

        else:

            log=np.concatenate([log,log2])

            

    models.append(copy.deepcopy(model0))

    print(i,'fold')

    plt.figure(figsize=(10,5))

    plt.plot(log[:,4])

    plt.plot(log[:,5])

    plt.ylim([0.50,0.60])

    plt.xlabel('epoch')

    plt.ylabel('kappa')

    plt.show()
def predicts(models,X,A):

    Y=np.zeros((len(X),4))

    for i in range(len(models)):

        Y+=models[i].predict([X,A],batch_size=2000)

    Y=Y/len(models)

    return Y
Yp=predicts(models,X_train,A_train)

print(quadratic_kappa(Y_train.argmax(axis=1).astype(np.int32), Yp.argmax(axis=1).astype(np.int32),sample_weight=W_train))

plt.hist([Y_train.argmax(axis=1),(Yp).argmax(axis=1)])#, stacked=True)

plt.show()
tmp=np.matmul(Yp,lossmat).argmax(axis=1).astype(np.int32)

print(quadratic_kappa(Y_train.argmax(axis=1).astype(np.int32), tmp ,sample_weight=W_train))

plt.hist([np.array(Y_train.argmax(axis=1).astype(np.int32),dtype=np.int32),tmp])

plt.show()
del X_train

test = pd.read_csv(dir+"test.csv")

del test['timestamp']

del test['type']

del test['world']

test=transform_df(test,nsplit=10)

Xt,At,Yt,Tt,IDt=make_set(test,fortest=True)

del test

gc.collect()

X_test,A_test = padding(Xt,At)

del Xt

gc.collect()

#y_pred = model0.predict([X_test,A_test],batch_size=2000)

y_pred = predicts(models,X_test,A_test)

ysub=np.matmul(y_pred,lossmat).argmax(axis=1).astype(np.int32)

sample_submission['accuracy_group'] = ysub

sample_submission.to_csv('submission.csv', index=False)

print(sample_submission['accuracy_group'].value_counts(normalize=True))
#import pickle

#with open('models.pickle', 'wb') as f:

#    pickle.dump(models, f)   

#with open('models.pickle', 'rb') as f:

#    models = pickle.load(f)