import os
import numpy as np
import pandas as pd
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.models import Model
from keras.layers import TimeDistributed,Dropout
import matplotlib.pyplot as plt
print(os.listdir("../input"))
EPOCHS=30
Batch_size=128
LSTM_NODES=256
max_features=58600
maxlen=500
embed_size=300
dropout_rate=0.2    #rate of the dropout layers
EMBEDDING_FILE="../input/bengali-word-embedding/bengali-word-embedding.txt"
# put words as dict indexes and vectors as words values
vocab_vector = {} 
with open(EMBEDDING_FILE,encoding='utf8') as f:  
    for line in f:
        values = line.rstrip().rsplit(' ')
        word_values = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        vocab_vector[word_values] = coefs
f.close()        
#print('Found %s word vectors.' % len(embeddings_index))
data = pd.read_csv("../input/dataset/dataset.csv")
text = data['doc']


label = data['label']
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index
num_words=len(word_index)+1
print('Found %s unique tokens.' %num_words)

#max_input_len=max(len(sen) for sen in sequence)
#print(max_input_len)
#print("Length of longest sentence in input:%g" % max_input_len)
#print(word_index)
#print('Found %s unique tokens.' % len(word_index))

encoder_input_sequences= pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')

print('shape of data tensor:',encoder_input_sequences.shape)
print(word_index["অধ্যাপক"])
### the label in our dataset was in string format,so convert them into int.and no need to tokenize them as thay are already in int format.
seq=[]
for string in label:
    li=list(string.split(" "))
    temp=[]
    for i in li:
        temp.append(int(i))
    seq.append(temp)
decoder_input_sequences=pad_sequences(seq,maxlen=maxlen,padding='post',truncating='post')    
print('Shape of label tensor:',decoder_input_sequences.shape)

#print(X)
#print('Shape of data tensor:', X.shape)
#print(X[172])
#print('Shape of label tensor:', Y.shape)
#print('filling pre=trained embeddings.....')
#num_words=min(max_features,len(word_index)+1) #50000 rakhlam

nb_words=min(max_features,len(word_index))
print(nb_words)
embedding_matrix=np.zeros((nb_words+1,embed_size))      #50000,embed size rakhlam


for embed_word,v in word_index.items():
    if v>=max_features:continue
    embedding_vector=vocab_vector.get(embed_word)
    
    
        #print(embedding_vector)
    #words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector
        
            
#create embedding layer
embedding_layer=Embedding(num_words,embed_size,weights=[embedding_matrix],input_length=maxlen)
decoder_targets_one_hot=np.zeros((len(text),maxlen,4),dtype='float32')          ##4 means 0,1, 2, 3 
decoder_targets_one_hot.shape

for j, d in enumerate(decoder_input_sequences):
    
    for t, word in enumerate(d):
       
        decoder_targets_one_hot[j, t, word] = 1
        
encoder_inputs_placeholder=Input(shape=(maxlen,))
x=embedding_layer(encoder_inputs_placeholder)

encoder=LSTM(LSTM_NODES,return_state=True)
encoder_outputs,h,c=encoder(x)
encoder_states=[h,c]
#decoder 
decoder_inputs_placeholder=Input(shape=(maxlen,))
decoder_embedding=Embedding(4,LSTM_NODES)
decoder_inputs_x=decoder_embedding(decoder_inputs_placeholder)
decoder_lstm=LSTM(LSTM_NODES,return_sequences=True,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs_x,initial_state=encoder_states)
decoder_dense=Dense(4,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)
model=Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder],decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
)
l=model.fit(
[encoder_input_sequences,decoder_input_sequences],
decoder_targets_one_hot,
batch_size=Batch_size,
epochs=EPOCHS,
validation_split=0.1,
)
encoder_model=Model(encoder_inputs_placeholder,encoder_states)
decoder_state_input_h=Input(shape=(LSTM_NODES,))
decoder_state_input_c=Input(shape=(LSTM_NODES,))
decoder_states_inputs=[decoder_state_input_h,decoder_state_input_c]
decoder_inputs_single=Input(shape=(1,))
decoder_inputs_single_x=decoder_embedding(decoder_inputs_single)
decoder_outputs,h,c=decoder_lstm(decoder_inputs_single_x,initial_state=decoder_inputs_inputs)
decoder_states=[h,c]
decoder_outputs=decoder_dense(decoder_outputs)
decoder_model=Model([decoder_inputs_single]+decoder_states_inputs,[decoder_outputs]+decoder_states)
#list all data in l
print(l.l.keys())
#summarize l for accuracy
plt.plot(l.l['accuracy'])
plt.plot(l.l['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper left')
plt.show()



