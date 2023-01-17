import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#import zipfile

#

#with zipfile.ZipFile("../input/quora-insincere-questions-classification/embeddings.zip") as zf:

    #print(zf.namelist())
test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')





print(len(test),len(train))

print(train.isna().sum())

train.head()
#import io



#import tqdm



#embedding_index = {}



#with zipfile.ZipFile("../input/quora-insincere-questions-classification/embeddings.zip") as zf:

    #with io.TextIOWrapper(zf.open("glove.840B.300d/glove.840B.300d.txt"), encoding="utf-8") as f:

       # for line in tqdm.tqdm(f):

          #  values = line.split(' ')# ".split(' ')" only for glove-840b-300d; for all other files, ".split()" works

          #  word = values[0]

           # coefs = np.asarray(values[1:],dtype = 'float32')

           # embedding_index[word] = coefs



#print('Loaded %s word vectors.' % len(embedding_index))

    
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score





from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



from tensorflow.keras.callbacks import EarlyStopping



def add_num_feat(df):

    

    df['len']               = df['question_text'].apply(lambda x: len(x))

    df['question_marks']    =  df['question_text'].str.count('\?')

    df['exclimat_marks']    =  df['question_text'].str.count('\!')

    df['dot_marks']         =  df['question_text'].str.count('\.')

    df['comma_marks']       =  df['question_text'].str.count('\,')

    df['capital_latters']   =  df['question_text'].str.count('[A-Z]')

    df['abbrivi_count']     =  df['question_text'].str.count('[A-Z]{2,}')

    

    return df





train = add_num_feat(train)

test  = add_num_feat(test)



train['len'].hist(bins = 100)

print('Mean len: {}\nSTD len: {}\nMin len: {}\nMax len: {}'.format(    np.mean(train['len']),

                                                                       np.std(train['len']),

                                                                       min(train['len']),

                                                                       max(train['len'])))



tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(train['question_text'])+list(test['question_text']))



vocab_size = len(tokenizer.word_index)+1

embedd_size = 300

max_len = 100



print('Vocab Size: {}\nEmbedding size: {}\nMax words: {}'.format(vocab_size,

                                                                 embedd_size, 

                                                                 max_len))
# Create random embedding matrix based on mean and std of glove



#all_embs = np.stack(list(embedding_index.values()))

#emb_mean,emb_std = all_embs.mean(), all_embs.std()

#nb_words = len(tokenizer.word_index)

#embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedd_size))
# replace random values with known values from glove



#for word, idx in tokenizer.word_index.items():

   # idx-=1

    #embedding_vector = embedding_index.get(word)

    #if embedding_vector is not None: 

                #embedding_matrix[idx] = embedding_vector

            

#print(len(embedding_index),len(embedding_matrix),len( tokenizer.word_index))
X = train['question_text'].values

y = train['target'].values



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,

                                               shuffle = True,

                                               random_state = 42)







X_train_tok = tokenizer.texts_to_sequences(X_train)

X_val_tok = tokenizer.texts_to_sequences(X_val)



X_train = pad_sequences(X_train_tok,maxlen=max_len)

X_val = pad_sequences(X_val_tok,maxlen=max_len)
model_seq = tf.keras.Sequential([

    # when add embedding remember to add embedding_size.shape[1],weight,trainable

    tf.keras.layers.Embedding(vocab_size,

                              64,

                              input_length = max_len),    

    tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True),input_shape=(max_len, 64)),

    tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64)),

    tf.keras.layers.Dense(64,activation = 'relu'),

    tf.keras.layers.Dense(1,activation = 'sigmoid')

        

])





model_seq.compile(loss = 'binary_crossentropy',

                  optimizer = 'adam',

                  metrics = ['accuracy'])



model_seq.fit(X_train,y_train,

                 batch_size=1024,

                 epochs=1,

                 validation_data=(X_val,y_val),

                 verbose=1)
from sklearn.metrics import f1_score 



def threshold_adj(y_true,y_pred):

    f1 = 0

    for i in np.arange(0.05,0.501,0.01):

        thresh = np.round(i, 2)

        f1_next = f1_score(y_true, np.where(y_pred > thresh,1,0))

        print('With threshold ',thresh,' f1 is ',f1_next )

        if f1_next > f1:

            f1 = f1_next

    

    return f1



pred_val = model_seq.predict(X_val,verbose=1,batch_size=1024)

threshold = threshold_adj(y_val,pred_val)

from sklearn.metrics import confusion_matrix



confusion_matrix(y_val,np.where(pred_val > threshold,1,0))







X_test = test['question_text'].values



X_test = tokenizer.texts_to_sequences(X_test)

X_test = pad_sequences(X_test,maxlen = max_len)



pred = model_seq.predict(X_test,verbose = 1,batch_size = 1024)



pred = np.where(pred > threshold,1,0)



test_qid = test['qid']

sub = pd.DataFrame({'qid':test_qid,'prediction':pred.reshape(pred.shape[0],)})

sub.to_csv('submission.csv',index = False)
#X = train.drop('target',axis = 1)

#y = train['target']



#X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,

                                               #shuffle = True,

                                               #random_state = 42)



#X_train_input_1 = X_train['question_text'].values

#X_train_input_2 = X_train.loc[:,'len':].values





#X_val_input_1 = X_val['question_text'].values

#X_val_input_2 = X_val.loc[:,'len':].values





#X_train_tok = tokenizer.texts_to_sequences(X_train_input_1)

#X_val_tok = tokenizer.texts_to_sequences(X_val_input_1)



#X_train = pad_sequences(X_train_tok,maxlen=max_len)

#X_val = pad_sequences(X_val_tok,maxlen=max_len)
#input1 = tf.keras.layers.Input(shape =(max_len,))#

#input2 = tf.keras.layers.Input(shape = (7,))



# when add embedding remember to add embedding_size.shape[1],weight,trainable

#embed       = tf.keras.layers.Embedding(len(tokenizer.word_index)+1,

                                        #8,

                                        #embedding_matrix.shape[1],

                                        #input_length = max_len,

                                        #weights = [embedding_matrix],

                                        #trainable    = True)(input1)



#blstm       = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embed)

#blstm_dense = tf.keras.layers.Dense(64,activation  = 'relu')(blstm)



#dense_num   = tf.keras.layers.Dense(128,activation = 'relu')(input2)

#dense2_num  = tf.keras.layers.Dense(64,activation  = 'relu')(dense_num)



#concat      = tf.keras.layers.concatenate([blstm_dense,dense2_num])



#dense3      = tf.keras.layers.Dense(32,activation = 'relu')(concat)

#dropout      = tf.keras.layers.Dropout(0.1)(dense3)



#final       = tf.keras.layers.Dense(1,activation  = 'sigmoid')(dropout)





#model = tf.keras.Model(inputs = [input1,input2],outputs = final)





#model.compile(loss      = 'binary_crossentropy',

              #optimizer = 'adam',

              #metrics   = ['accuracy'])





#tf.keras.utils.plot_model( 

    #model, to_file='model.png',

    #show_shapes=True, show_layer_names=True,

    #rankdir='TB', expand_nested=False, dpi=96

#)
#model.fit([X_train,X_train_input_2],y_train,

          #batch_size=1024,

          #epochs=1,

          #validation_data=([X_val,X_val_input_2]),

          #verbose=1)



#pred_val = model.predict([X_val,X_val_input_2],verbose=1,batch_size=1024)

#threshold = threshold_adj(y_val,pred_val)