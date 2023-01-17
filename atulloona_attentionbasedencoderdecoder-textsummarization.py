%config IPCompleter.greedy=True

from numpy.random import seed
seed(1)

modelLocation="C:\\Summary\\"
##############
import numpy as np
import os
import pandas as pd
import re
############
import gensim as gs
import pandas as pd
import numpy as np
import scipy as sc
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import logging
import re
from collections import Counter
#######################
from numpy.random import seed
seed(1)
from sklearn.model_selection import train_test_split as tts
import logging
import matplotlib.pyplot as plt
import pandas as pd
#import pydot
import tensorflow as tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Dropout,Input,Activation,Add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Embedding,Bidirectional,dot,TimeDistributed
#from tensorflow.keras.layers.advanced_activations import LeakyReLU,PReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers.embedding import Embedding
############################
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot

totalfiles = 1000
# embedding_size = 64
# hidden_units = 50
input_dict_size =  100

output_dict_size = 100
CNN_data="D:\\Text Summarisation Encoder Decoder\\cnn\\"
daily_data="D:\\Text Summarisation Encoder Decoder\\cnn\\" 
CNN_data="../input/cnn-summary/cnn/"
daily_data="../input/cnn-summary/cnn/" 

datasets={"cnn":CNN_data,"dailymail":daily_data}
data_categories=["training","validation","test"]
data={"articles":[],"summaries":[]}

def parsetext(dire,category,filename):
    with open("%s%s/%s"%(dire,category,filename),'r',encoding="Latin-1") as readin:
        text=readin.read()
    return text.lower()

def cleantext(text):
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"\."," ",text)
    text=re.sub(r","," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)
    return text

def printArticlesum(k):
    print("---------------------original sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["articles"][k])
    print("----------------------Summary sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["summaries"][k])
    return 0

def load_data(dire,category):
    """dataname refers to either training, test or validation"""
    filesnames = [];
    for dirs,subdr, files in os.walk(dire+category):
        filesnames=files
    return filesnames

filenames=load_data(datasets["cnn"],data_categories[0])
print("Total files in category training",len(filenames))

for k in range(len(filenames[:totalfiles])):
        firstname = os.path.splitext(filenames[k])[0];
        extention =  os.path.splitext(filenames[k])[1];
        if(firstname == '00465603227f7f56fcd37e10f4cd44e57d7647d8'):
            continue
        if os.path.splitext(filenames[k])[1] == '.sent':
            try:
                data["articles"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])))
                name = firstname + '.summ';
                data["summaries"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%name)))
            except Exception as e:
                data["articles"].append("Could not read")
        if (os.path.splitext(filenames[k])[1] == '.summ'):
            try:
                data["summaries"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])))
                name = firstname + '.sent';
                data["articles"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%name)))
            except Exception as e:
                data["summaries"].append("Could not read")

print("Total summary",len(data["summaries"]))
print("Total Articles",len(data["articles"]))
# Creating a tokenizer
tokenizer = Tokenizer(lower=True)
#print(type(data["summaries"]))
combineddata = []
for sent in data["articles"]: 
    combineddata.append(sent) 
for sent in data["summaries"] : 
    combineddata.append(sent) 

words = set(w for sent in combineddata for w in sent.split())

tokenizer.fit_on_texts(words)

encoded_docs_articles = [tokenizer.texts_to_sequences(sent.split()) for sent in data["articles"]]
encoded_docs_summary = [tokenizer.texts_to_sequences(sent.split()) for sent in data["summaries"]]

# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

output_dict_size = vocab_size = len(tokenizer.word_index)
print("Total no. of words in vocabulary is", vocab_size)

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(index) for index in list_of_indices]
    return(words)

# Creating texts 
#my_texts = list(map(sequence_to_text, encoded_docs_articles[0]))
#print(type(encoded_docs_articles))
#print(data["articles"][176])
#print(encoded_docs_articles[176])
encoded_docs_articles = np.delete(encoded_docs_articles, 452)
encoded_docs_summary = np.delete(encoded_docs_summary, 452)

for k in range(0,len(encoded_docs_articles)):
    if len(encoded_docs_articles[k]) > 0:
        encoded_docs_articles[k] = np.concatenate( encoded_docs_articles[k], axis=0 )
    else:
        print(k)
for k in range(0,len(encoded_docs_summary)):
    if  len(encoded_docs_summary[k]) > 0:
        encoded_docs_summary[k] = np.concatenate( encoded_docs_summary[k], axis=0 )
    else:
        print(k)
def max_len(data):
    lenk=[]
    for k in data:
        lenk.append(len(k))
    return  np.max(lenk) 

max_length_article = max_len(data["articles"])
max_length_summary = max_len(data["summaries"])

INPUT_LENGTH = max_length_article = 200
OUTPUT_LENGTH = max_length_summary = 60

print("Total no. words in fed to encoder is in one sample", INPUT_LENGTH)
print("Total no. words in fed to decoder is in one sample", OUTPUT_LENGTH)


padded_docs_article = pad_sequences(encoded_docs_articles, maxlen=max_length_article, padding='post')
padded_docs_summary = pad_sequences(encoded_docs_summary, maxlen=max_length_summary, padding='post')

rge = 400
padded_docs_article = padded_docs_article[0:rge]
padded_docs_summary = padded_docs_summary[0:rge]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#######################model params###########################
epochs = 500
embedding_size = 64
hidden_units = 64
############################helpers###########################
def addfirst(seq):
    #print(seq)
    arr = []
    for val in seq:
        ary = np.zeros(output_dict_size+1)
        ary[val] = 1
        arr.append(ary)
    #print(arr)    
    return arr
def addStart(seq):
    #print(seq)
    #print(tokenizer.texts_to_sequences("start".split())[0])
    retseq = np.insert(seq,0,tokenizer.texts_to_sequences("start".split())[0], axis = 0)
    retseq = np.delete(retseq, -1)
    #print(retseq)
    return retseq

###############################################################
def encoder_decoder():  
    print('Encoder_Decoder LSTM...')
    """__encoder___""" 
    x_train,x_test,y_train,y_test=tts(padded_docs_article,padded_docs_summary,test_size=0.3)

    y_train_start = np.array(list(map(addStart,y_train)))
    #print(y_train)

    y_train_out = np.array(list(map(addfirst,y_train)))

    #x_test = np.array(list(map(addfirst,x_test)))
    #y_test_startadded =y_test# np.array(list(map(addStart,y_test)))

    y_test_out = np.array(list(map(addfirst,y_test)))

    print("Training input samples shape in encoder ", x_train.shape)
    print("Training input samples shape in decoder ",y_train.shape) #(80, 9576)vocab_size
    print("Test samples shape in encoder ", x_test.shape)
    print("Test samples shape in decoder ",y_test.shape)
#     encoder_input = Input(shape=(INPUT_LENGTH,vocab_size))
#     decoder_input = Input(shape=(OUTPUT_LENGTH,vocab_size))

    encoder_input = Input(shape=(INPUT_LENGTH,))
    decoder_input = Input(shape=(OUTPUT_LENGTH,))

    print("encoder input",encoder_input)
    input_emb = Embedding(vocab_size+10, embedding_size, input_length=INPUT_LENGTH,mask_zero=True)(encoder_input) 

    encoder, encoder_hstate, encoder_context = LSTM(hidden_units, return_sequences=True,return_state=True, unroll=True)(input_emb)
    encoder_last = encoder[:,-1,:]

    # all output state, last hidden state and all context state,
    #print('encoder shapes', encoder.shape,encoder_hstate.shape,encoder_context.shape,encoder_last.shape)

    """____decoder___"""
    #outputlen = OUTPUT_LENGTH;
    decoder = Embedding(output_dict_size, embedding_size, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
    decoder = LSTM(hidden_units, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_hstate, encoder_context])
#    print('decoder shapes', decoder.shape)

    attention = dot([decoder, encoder], axes=[2, 2]) #A[j][i] = softmax( D[j] * E[i] ) # softmax by row
#    print('attention dot', attention) #20*100
    attention = Activation('softmax', name='attention')(attention) # dense 64 
#    print('attention softmax', attention)

    context = dot([attention, encoder], axes=[2,1])

    #C[j] = sum( [A[j][i] * E[i] for i range(0, INPUT_LENGTH)] )
#    print('context dot', context)

    decoder_combined_context = concatenate([context, decoder])
#    print('decoder_combined_context', decoder_combined_context)

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
#     print("output tanh layer",output.shape)
    output = TimeDistributed(Dense(output_dict_size+1, activation="softmax"))(output)
#     print(output.shape)
#     print('output softmax layer', output)
#     print('attention shape', attention.shape)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=[x_train, y_train_start], y=[y_train_out],
              validation_data=([x_test, y_test], [y_test_out]),
              verbose=2, batch_size=1, epochs=epochs)
    return model 
def generate(text):
        #print(text)
        encoder_input = np.reshape(np.array(text),(1,INPUT_LENGTH)) 
        #print(encoder_input.shape)
        decoder_input = np.zeros(shape=(OUTPUT_LENGTH-1,)) #(100,20 )
        print(tokenizer.texts_to_sequences("start".split())[0])
        decoder_input = np.insert(decoder_input,0,tokenizer.texts_to_sequences("start".split())[0], axis = 0)
        
        decoder_input = np.reshape(decoder_input,(1,OUTPUT_LENGTH)) 
        sent = " "
        #decoder_input = np.concatenate( decoder_input, axis=0 )
        #print(decoder_input.shape)
        #print([encoder_input, decoder_input])
        #print([encoder_input, decoder_input].shape)
        for i in range(1, OUTPUT_LENGTH):
            output = model.predict([encoder_input, decoder_input])
            #print(output)
            #print(output.shape)
            #print( output[0][i-1].shape)
            sentindex = output[0][i-1].argmax()
            #print(sentindex)
            sent = sent + sequence_to_text([sentindex])
            #print(output.argmax(axis=2) , i-1)
            #sentindex = output.argmax(axis=2)[0][i-1]
            #print(sentindex)
            #print(sequence_to_text([sentindex]))
            decoder_input[0][i] = sentindex
            #print(decoder_input)
        return sent
def decode(decoding, sequence):
        text = ''
        for i in sequence:
            if i == 0:
                break
            text += output_decoding[i]
        return text

"""___pred____"""
def comparePred(index):
    pred=trained_model.predict([np.reshape(train_data["article"][index],(1,en_shape[0],emb_size_all)),np.reshape(train_data["summaries"][index],(1,de_shape[0],emb_size_all))])
    return pred


"""____generate summary from vectors and remove padding words___"""
def generateText(SentOfVecs):
    SentOfVecs=np.reshape(SentOfVecs,de_shape)
    kk=""
    for k in SentOfVecs:
        kk=kk+((getWord(k)[0]+" ") if getWord(k)[1]>0.2 else "")
    return kk

"""___generate summary vectors___"""
model = encoder_decoder()
model.save("Summarymodel_Encoder.h5")
print("Saved model to disk")
print(generate(padded_docs_article[30]))
print(data["summaries"][30])

plt.plot(model.history.history['accuracy'][:])
plt.plot(model.history.history['val_accuracy'][:])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(index) for index in list_of_indices]
    return(words)

def generate(text):
        #print(text)
        encoder_input = np.reshape(np.array(text),(1,INPUT_LENGTH)) 
        #print(encoder_input.shape)
        decoder_input = np.zeros(shape=(OUTPUT_LENGTH-1,)) #(100,20 )
        print(tokenizer.texts_to_sequences("start".split())[0])
        decoder_input = np.insert(decoder_input,0,tokenizer.texts_to_sequences("start".split())[0], axis = 0)
        
        decoder_input = np.reshape(decoder_input,(1,OUTPUT_LENGTH)) 
        sent = " "
        #decoder_input = np.concatenate( decoder_input, axis=0 )
        #print(decoder_input.shape)
        #print([encoder_input, decoder_input])
        #print([encoder_input, decoder_input].shape)
        for i in range(1, OUTPUT_LENGTH):
            output = model.predict([encoder_input, decoder_input])
            #print(output)
            #print(output.shape)
            #print( output[0][i-1].shape)
            sentindex = output[0][i-1].argmax()
            #print(sentindex)
            if sentindex != 0:
                sent = sent +" " + sequence_to_text([sentindex])[0]
            #print(output.argmax(axis=2) , i-1)
            #sentindex = output.argmax(axis=2)[0][i-1]
            #print(sentindex)
            #print(sequence_to_text([sentindex]))
            decoder_input[0][i] = sentindex
            #print(decoder_input)
        return sent
def decode(decoding, sequence):
        text = ''
        for i in sequence:
            if i == 0:
                break
            text += output_decoding[i]
        return text


print("Saved model to disk")
print(generate(padded_docs_article[30]))
print(data["summaries"][30])