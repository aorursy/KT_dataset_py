# source text format . For example if below is the input sequence, we will feed the data to the model as given in the 
#next line. We are going to use davincicode.txt file as the input which has more than 100K sentenses
data = 'Jack and Jill went up the hill To fetch a pail of water Jack fell down and broke his crown'
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
book_text=open('../input/davincicodetxt/davincicode.txt',encoding='UTF-8').read()
len(book_text)
sequences=list()
for line in book_text.split('.'):
    sequences.append(line)
len(sequences)
sequences[2]
def clean_str(string):
  """
  String cleaning before vectorization
  """
  try:    
    string = re.sub(r'^https?:\/\/<>.*[\r\n]*', '', string, flags=re.MULTILINE)
    string = re.sub(r"[^A-Za-z]", " ", string)         
    words = string.strip().lower().split()    
    words = [w for w in words if len(w)>=1]
    if len(words)>1:
        return " ".join(words)	
    else:
        return 'NA'
    
  except:
    return ""
cleaned_seq=list() 
for line in sequences:
    cleaned_seq.append(clean_str(line))
    
cleaned_seq[2]
len(cleaned_seq)
cleaned_seq2=list()
for line in cleaned_seq:
    if line!='NA':
        cleaned_seq2.append(line)


len(cleaned_seq2)
cleaned_seq2
token=Tokenizer()
token.fit_on_texts(cleaned_seq2)
token.word_index
# determine the vocabulary size
vocab_size = len(token.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
encoded=token.texts_to_sequences(cleaned_seq2)
encoded
len(encoded)
sequence_list=list()
for line in encoded:
    for i in range(1,len(line)):
        sequence_list.append(line[:i+1])
        
        
print('Total Sequences_list:',len(sequence_list))
sequence_list
max_length=max([len(seq) for seq in sequence_list])
print ('maximum sequence length is', max_length)
from keras.preprocessing.sequence import pad_sequences
sequence_padded_list=pad_sequences(sequence_list,maxlen=max_length,padding='pre')
sequence_padded_list

sequence_padded_list=np.array(sequence_padded_list)


sequence_padded_list.shape
# Split the input and output data
X,y= sequence_padded_list[:,:-1],sequence_padded_list[:,-1]
X
X.shape
y
y.shape
from keras.utils import to_categorical

vocab_size
y_cat=to_categorical(y,num_classes=vocab_size)
y_cat
y_cat.shape
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout,Embedding,LSTM
from keras.optimizers import RMSprop
# instantiating the model in the strategy scope creates the model on the TPU
#with tpu_strategy.scope():
model=Sequential()
model.add(Embedding(vocab_size,40,input_length=max_length-1))
model.add(LSTM(300,return_sequences=True))
model.add(LSTM(200))
model.add(Dense(vocab_size,activation='softmax'))
optim=RMSprop(lr=0.07)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




model.summary()

model.fit(X,y_cat,epochs=110,batch_size=100,verbose=2)
model.save('keras_next_word_model_StakckedLSTM.h5')
import keras
model1=keras.models.load_model('./keras_next_word_model_StakckedLSTM.h5')


print(generate_seq(model1,token,max_length-1,'pagan',10))
# Generate sequence from language model

def generate_seq(model,token,max_length,seed_text,n_words):
    in_text=seed_text
    
    # generate the fixed number of words given by n_words in the input to the function
    for _ in range(n_words):
        # encode the text as integeer
        encoded_text=token.texts_to_sequences([in_text])[0]
        #prepad same as we did before training
        encoded_text=pad_sequences([encoded_text],maxlen=max_length,padding='pre')
        #predict the probabilities of the word
        y_pred=model.predict_classes(encoded_text,verbose=0)
        
        #map the index to the word
        
        outword=''
        for word, index in token.word_index.items():
            if index == y_pred:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    
    return in_text