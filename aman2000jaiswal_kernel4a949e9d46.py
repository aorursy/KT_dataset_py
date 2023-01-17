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
import tensorflow as tf

import keras

import keras.layers as L

from keras.layers import Dense,Embedding,SimpleRNN,Flatten,InputLayer

from keras.models import Sequential

from keras.layers import LSTM

import nltk

def jaccard(ox,oy):

    evaluator=0

    count=0

    for i in range(len(ox)):

        x=nltk.word_tokenize(ox[i])

        y=nltk.word_tokenize(oy[i])

#         x=ox

#         y=oy

        xs=set(map(lambda i:i.lower(),x))

        ys=set(map(lambda i:i.lower(),y))

        c=xs.intersection(ys)

        count+=1

        evaluator+=(len(c)) / (len(xs) + len(ys) - len(c))

    evaluator=evaluator/count                           

    return evaluator    

train_org=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_org.head()

train=train_org.dropna(axis=0)

iii=train.text.values

document=' '.join(list(iii))

document+=" padpad"

document+=' unkunk'

pad='padpad'

unk='unkunk'

# adding vocab

tokenize_words=nltk.word_tokenize(document)

from collections import Counter

counted_vocab=Counter(tokenize_words)

vocab=counted_vocab.keys()

token_id={}

for num,i in enumerate(vocab):

    token_id[i]=num

n_tokens=len(vocab)

max_len=50

def to_matrix(batch,token_id=token_id):

    mat=[]

    for i in batch:

        mat1=[]

        for token in list(nltk.word_tokenize(i)):

            st=str(token)

            if(st in token_id.keys()):

                mat1.append(token_id[st])

            else:

                mat1.append(token_id[unk])

        mat.append(mat1)

    return mat

def pad_to_mat(matrix,pad_token=token_id[pad],max_len=max_len):

    zero=np.zeros((len(matrix),max_len)) + pad_token

    for i in range (len(matrix)):

        for num,j in enumerate(matrix[i]):

            if(num<max_len):

                zero[i][num]=j

    return zero

def maskvalue(xl,token_id=token_id):

    mask_array=[]

    for i in range(len(xl)):

        mask_array1=[]

        for j in range(len(xl[i])):

            if(xl[i][j]==token_id[pad] or xl[i][j]==token_id[unk]):

                mask_array1.append(0)

            else:

                mask_array1.append(1)

        mask_array.append(mask_array1)        

    return np.array(mask_array) 

def preprocessing(batch,max_len=max_len):

    mat1=to_matrix(batch)

    mat2=pad_to_mat(mat1,max_len=max_len)

    mat3=maskvalue(mat2)

    return mat1,mat2,mat3
def output(batch_X,batch_y,max_len=max_len):

    _,ip1,_=preprocessing(batch_X,max_len=max_len)

    out1,_,_=preprocessing(batch_y,max_len=max_len)

    res=[]

    for i in range(len(ip1)):

        res1=[]

        for j in range(len(ip1[i])):

            if(ip1[i][j] in out1[i] or ip1[i][j]==token_id[pad]):

                res1.append(1)

            else:

                res1.append(0)

        res.append(res1)

    ret_res=np.array(keras.backend.one_hot(res,num_classes=2))

    return ret_res

def real_values(model,val_x,mask=None,token_id=token_id):

    prediction=model.predict(val_x)

    true_value=[]

    for i in prediction:

        true_value1=[]

        for j in i:

            true_value1.append(np.argmax(j))

        true_value.append(true_value1)      

    pred_token_id=[]

    for i in range(len(true_value)):

        pred_token1_id=[]

        for j in range(len(true_value[0])):

            try:

                if(mask[i][j]==1):

                    if(val_x[i][j]!=token_id[pad] and val_x[i][j]!=token_id[unk]):

                        pred_token1_id.append(val_x[i][j])

            except:

                if(val_x[i][j]!=token_id[pad] and val_x[i][j]!=token_id[unk]):

                    pred_token1_id.append(val_x[i][j])

                

        pred_token_id.append(pred_token1_id)

    int_id=[]

    for i in pred_token_id:

        int_id1=[]

        for j in i:

            int_id1.append(int(j))

        int_id.append(int_id1)

    pred_token_id=int_id        

    keylist=list(token_id.keys())

    ret_pred_token=[]

    for i in range(len(pred_token_id)):

        pred_token1=[]

        for j in range(len(pred_token_id[i])):

            pred_token1.append(keylist[pred_token_id[i][j]])

        ret_pred_token.append(pred_token1)

    return ret_pred_token
test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
import numpy

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

# fix random seed for reproducibility



numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest

top_words = n_tokens

X_train,_,_=preprocessing(train.text.values)

y_train=output(train.text.values,train.selected_text.values)



# y_test=output(X_pos_valid.values,y_pos_valid.values)

# truncate and pad input sequences

max_review_length = 50

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length,value=token_id[pad], padding='post')

# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length,value=token_id[pad], padding='post')

# create the model

embedding_vecor_length = 300

pos_train=train[train.sentiment=='positive']

neg_train=train[train.sentiment=='negative']

neu_train=train[train.sentiment=='neutral']

from sklearn.model_selection import train_test_split

X_pos_train,X_pos_valid,y_pos_train,y_pos_valid=train_test_split(pos_train.text,pos_train.selected_text,random_state=42)

X_neg_train,X_neg_valid,y_neg_train,y_neg_valid=train_test_split(neg_train.text,neg_train.selected_text,random_state=42)

X_neu_train,X_neu_valid,y_neu_train,y_neu_valid=train_test_split(neu_train.text,neu_train.selected_text,random_state=42)
pos_test=test[test.sentiment=='positive']

neg_test=test[test.sentiment=='negative']

neu_test=test[test.sentiment=='neutral']
def training_model(X,X1,y,y1,X2,max_len=max_len):

    top_words = n_tokens

    X_train,_,_=preprocessing(X.values,max_len=max_len)

    y_train=output(X.values,y.values,max_len=max_len)

    

    X_valid,_,_=preprocessing(X1.values,max_len=max_len)

    y_valid=output(X1.values,y1.values,max_len)

    

    X_test,_,_=preprocessing(X2.text.values,max_len=max_len)

    

    max_review_length = max_len

    

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length,value=token_id[pad], padding='post')

    

    X_valid = sequence.pad_sequences(X_valid, maxlen=max_review_length,value=token_id[pad], padding='post')

    

    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length,value=token_id[pad], padding='post')

    

    embedding_vecor_length = 100

    rnn=300

    tf.keras.backend.clear_session()

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(max_review_length,)))

    model.add(tf.keras.layers.Masking(mask_value=token_id[pad]))

    model.add(tf.keras.layers.Embedding(top_words, embedding_vecor_length))

    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.LSTM(rnn,return_sequences=True))

    model.add(tf.keras.layers.Dropout(0.4))

    step_wise_dense=tf.keras.layers.Dense(100,input_shape=(rnn,),activation='relu')

    step_wise_dense1=tf.keras.layers.TimeDistributed(step_wise_dense)

    model.add(step_wise_dense1)

    model.add(tf.keras.layers.Dense(300,activation='relu',input_shape=(max_len,)))

    model.add(tf.keras.layers.Dense(150, activation='elu'))

    model.add(tf.keras.layers.Dropout(0.4))

#     model.add(Dense(150, activation='elu'))

#     model.add(Dropout(0.4))

#     model.add(Dense(50, activation='elu'))

#     model.add(Dropout(0.4))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    model.fit(X_train, y_train, epochs=20, batch_size=32,validation_data=[X_valid,y_valid])

    # Final evaluation of the model

    scores = model.evaluate(X_train, y_train, verbose=0)

    scores1 = model.evaluate(X_train, y_train, verbose=0)

          

    print("Accuracy: %.2f%%" % (scores[1]*100))

    print("1 Accuracy: %.2f%%" % (scores1[1]*100))

          

    predpred=real_values(model,X_train,mask=maskvalue(X_train),token_id=token_id)

    predp=[]

    for i in predpred:

        predp1=' '.join(i)

        predp.append(predp1)

    df=pd.DataFrame(predp,columns=['selected_text'])

    print("train jaccard : ",jaccard(y.values,df.selected_text.values))

          

    predpred=real_values(model,X_valid,mask=maskvalue(X_valid),token_id=token_id)

    predp=[]

    for i in predpred:

        predp1=' '.join(i)

        predp.append(predp1)

    df=pd.DataFrame(predp,columns=['selected_text'])

    print("train jaccard : ",jaccard(y1.values,df.selected_text.values))

    

    tpredpred=real_values(model,X_test,mask=maskvalue(X_test),token_id=token_id)

    tpredp=[]

    for i in tpredpred:

        tpredp1=' '.join(i)

        tpredp.append(tpredp1)

#     print(len(tpredp))

    dft=pd.DataFrame(tpredp,columns=['selected_text'],index=X2.index)

#     print(dft.shape)

    dftt=pd.concat([X2.textID,dft],axis=1)

    print(dftt.shape)

    print(X2.shape)

#     assert dft.shape[0]==X2.shape[0]

    return dftt            

df1=training_model(X_pos_train,X_pos_valid,y_pos_train,y_pos_valid,pos_test,max_len=35)
df2=training_model(X_neg_train,X_neg_valid,y_neg_train,y_neg_valid,neg_test,max_len=35)
df3=training_model(X_neu_train,X_neu_valid,y_neu_train,y_neu_valid,neu_test,max_len=50)
final_df=pd.concat([df1,df2,df3],axis=0)

final_df.shape
final_df.to_csv('submission.csv',index=False)