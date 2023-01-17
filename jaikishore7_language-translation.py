import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.enable_eager_execution()
import random
import nltk
lines=open('../input/deu.txt', encoding='utf-8', errors='ignore').read().split('\n')
pairs = [line.split('\t') for line in  lines]
pairs=pairs[0:-1]
questions=[]
answers=[]
for i in range(0, len(pairs)):
    questions.append(pairs[i][1])
    answers.append(pairs[i][0])
data=questions+answers
for i in range(0,len(data)):
    data[i]=data[i].lower()
import re
for i in range(0,len(data)):
    data[i]=re.sub(r'\d+','',data[i])
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer(r'\w+')
for i in range(0,len(data)):
    data[i]=tokenizer.tokenize(data[i])
ques=[]
ans=[]
for i in range(0,len(data)):
    if i<len(questions):
        ques.append(data[i][:13])
    else:
        ans.append(data[i][:13])
ques = ques[:8000]
ans = ans[:8000]
for i in range(len(ques)):
    ques[i] =  (9-len(ques[i])) * ['<pad>'] + ques[i]
    ans[i] = ['<start>'] + ans[i] + ['<end>'] + (7 - len(ans[i])) * ['<pad>']
from gensim.models import Word2Vec

w2v_enc=Word2Vec(sentences=ques,min_count=1,size=50,iter=50,window = 3)
w2v_dec=Word2Vec(sentences=ans,min_count=1,size=50,iter=50,window=3)
vocab_dec=w2v_dec.wv.vocab

vocab_dec=list(vocab_dec)
int_to_vocab_dec={}
for i in range(0,len(vocab_dec)):
    int_to_vocab_dec[i]=vocab_dec[i]
vocab_to_int_dec={}
for key,value in int_to_vocab_dec.items():
    vocab_to_int_dec[value]=key
vocab_enc=w2v_enc.wv.vocab

vocab_enc=list(vocab_enc)
int_to_vocab_enc={}
for i in range(0,len(vocab_enc)):
    int_to_vocab_enc[i]=vocab_enc[i]
vocab_to_int_enc={}
for key,value in int_to_vocab_enc.items():
    vocab_to_int_enc[value]=key
len(vocab_to_int_dec)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ques,ans,test_size = 0.1,random_state = 1234,shuffle = True)
dec_inp_train = np.zeros([len(y_train),9,50])
dec_inp_test = np.zeros([len(y_test),9,50])
for i in range(len(y_train)):
    temp = y_train[i].copy()
    try:
        temp[temp.index('<end>')] = '<pad>'
    except ValueError:
        pass
    y_train[i] = y_train[i][1:] + ['<pad>']
    
    dec_inp_train[i] = w2v_dec.wv[temp]
    x_train[i] = w2v_enc.wv[x_train[i]]

for i in range(len(y_test)):
    temp = y_test[i].copy()
    try:
        temp[temp.index('<end>')] = '<pad>'
    except ValueError:
        pass
    y_test[i] = y_test[i][1:] + ['<pad>']
    
    dec_inp_test[i] = w2v_dec.wv[temp]
    x_test[i] = w2v_enc.wv[x_test[i]]
class attention(tf.keras.Model):
    def __init__(self):
        super(tf.keras.Model,self).__init__()
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128,return_sequences=True,return_state=True))
        self.decoder = tf.keras.layers.CuDNNLSTM(256,return_state=True)
        self.dense = tf.keras.layers.Dense(400,activation='relu')
        self.out = tf.keras.layers.Dense(1915)
        self.attention_dense = tf.keras.layers.Dense(1,activation='tanh')
        self.attention_softmax = tf.keras.layers.Dense(1,activation='softmax')
        
            
    def encoder_func(self,inp):
        values,ht1,ct1,ht2,ct2 = self.encoder(inp)
        ht1 = tf.reshape(ht1[-1],shape=[1,128])
        ht2 = tf.reshape(ht2[-1],shape=[1,128])
        ct1 = tf.reshape(ct1[-1],shape=[1,128])
        ct2 = tf.reshape(ct2[-1],shape=[1,128])
        
        ht = tf.concat([ht1,ht2],axis=1)
        ct = tf.concat([ct1,ct2],axis=1)
        
        return values,ht,ct
        
    
    def decoder_func(self,enc_inp,dec_input = None):
        deco_out = tf.convert_to_tensor(w2v_dec['<start>'],dtype=tf.float32)
        deco_out = tf.reshape(deco_out,shape=[1,1,50])
        count = 0
        value = 0
        predictions = tf.zeros([1,1915])
        
        encoder_states,h_t,c_t = self.encoder_func(enc_inp)
        
        if dec_input != None:
            for i in range(16):
                for j in range(9):
                    dec_inp = self.attention_func(h_t,dec_input[i][j],encoder_states[i])
                    value,h_t,c_t = self.decoder(dec_inp,initial_state= [h_t,c_t])
                    value = self.dense(value)
                    value = self.out(value)
                    predictions = tf.concat([predictions,value],axis=0)
            predictions = predictions[1:]
            predictions = tf.reshape(predictions,[-1,9,1915])
            return predictions
        else:
            sentence = []
            while count < 9 and int_to_vocab_dec[value] != '<end>':
                dec_inp = self.attention_func(h_t,deco_out,encoder_states[0])
                value,h_t,c_t = self.decoder(deco_out,initial_state = [h_t,c_t])
                value = self.dense(value)
                value = self.out(value)
                value = tf.nn.softmax(value)
                value = random.choice(np.argsort(value[0])[-3:])
                sentence.append(int_to_vocab_dec[value])
                count += 1
                deco_out = tf.convert_to_tensor(w2v_dec[int_to_vocab_dec[value]])
                deco_out = tf.reshape(deco_out,shape=[1,1,50])
            return sentence[:-1]
                
    def attention_func(self,dec_h_t,decoder_out,enc_state):
        
        temp = tf.zeros([1,512])
        for i in range(9):
            enc_statee=enc_state[i]
            enc_statee=tf.reshape(enc_statee,(1,-1))
            temp1=tf.concat([enc_statee,dec_h_t],axis=1)
            temp=tf.concat([temp,temp1],axis=0)
        temp=temp[1:]
        
        attention_weights = self.attention_dense(temp)
        attention_weights = self.attention_softmax(attention_weights)
        
        context_vector = tf.matmul(tf.transpose(enc_state),attention_weights)
        decoder_out=tf.reshape(decoder_out,(-1,1))
        attention_context = tf.concat([decoder_out,context_vector],axis=0)
        attention_context = tf.reshape(attention_context,(1,1,-1))

        return attention_context     
model = attention()
optimzer = tf.train.RMSPropOptimizer(learning_rate=0.01)
def loss_fun(x,y,z):
    with tf.GradientTape() as t:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=z,logits=model.decoder_func(x,y)))
        grads = t.gradient(loss,model.variables)
        optimzer.apply_gradients(zip(grads,model.variables))
    return loss
for epoch in range(3):
    i = 0
    while i < len(x_train):
        a = np.array(x_train[i:i+16])
        b = np.array(dec_inp_train[i:i+16])
        temp = y_train[i:i+16]

        c = np.zeros([16,9,1915])

        for k in range(16):
            for j in range(9):
                c[k][j][vocab_to_int_dec[temp[k][j]]] = 1

        los = loss_fun(tf.convert_to_tensor(a,dtype=tf.float32),tf.convert_to_tensor(b,dtype=tf.float32),c)
        
        i = i + 16
        
        if i % 128 == 0:
            score = 0
            test_temp_enc,test_temp_dec = zip(*random.sample(list(zip(x_test, y_test)), 20))
            for m in range(20):
                prediction_sent = model.decoder_func(tf.convert_to_tensor(test_temp_enc[m].reshape([1,9,50]),dtype=tf.float32))
                actual_sent = test_temp_dec[m][:test_temp_dec[m].index('<end>')]
                score += nltk.translate.bleu_score.sentence_bleu([actual_sent],prediction_sent)
            print("bleu score when i is: ",i, " is: ",score/20)
