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
from sklearn.model_selection import train_test_split

#print(os.listdir("../input"))
Batch_Size=32
max_features=58600
maxlen=200
embed_size=300

#n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.
LSTM_NODES=128

EMBEDDING_FILE='../input/bengali-word-embedding/bengali-word-embedding.txt'
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
data = pd.read_csv('../input/dataset/dataset.csv')
text = data['doc']
print(text[1])
#label=data['label']
#print(label[1])
data['label']=data['label'].apply(lambda x : ' sos ' +  x  + ' eos')
label=data['label']
print(label[1])
   
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index
num_words=len(word_index)+1
print('Found %s unique tokens.' %num_words)
#max_input_len = max(len(sen) for sen in sequence)
#print(max_input_len)

#ax_input_len=max(len(sen) for sen in sequence)
#print(max_input_len)
#print("Length of longest sentence in input:%g" % max_input_len)
#print(word_index)
#print('Found %s unique tokens.' % len(word_index))

encoder_input_sequences= pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')

print('shape of data tensor:',encoder_input_sequences.shape)
#print(word_index["অধ্যাপক"])
### the label in our dataset was in string format,so convert them into int.and no need to tokenize them as thay are already in int format.
t = Tokenizer(num_words=max_features)
t.fit_on_texts(label)
seq = t.texts_to_sequences(label)
word_index1 = t.word_index
num_words_output=len(word_index1)+1

#max_out_len = max(len(sen) for sen in seq)
decoder_input_sequences=pad_sequences(seq,maxlen=maxlen,padding='post',truncating='post')    
print('Shape of label tensor:',decoder_input_sequences.shape)


encoder_input_sequences_train,encoder_input_sequences_test,decoder_input_sequences_train,decoder_input_sequences_test=train_test_split(encoder_input_sequences,decoder_input_sequences,test_size=0.33,random_state=42)        

nb_words=min(max_features,len(word_index))
#print(nb_words)
embedding_matrix=np.zeros((nb_words+1,embed_size))      #50000,embed size rakhlam


for embed_word,v in word_index.items():
    if v>=max_features:continue
    embedding_vector=vocab_vector.get(embed_word)
    
    
        #print(embedding_vector)
        
    #words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector 

embedding_layer=Embedding(num_words,embed_size,weights=[embedding_matrix],input_length=maxlen)
decoder_targets_one_hot = np.zeros((
        len(text),
        maxlen,
        num_words_output
        
    ),
    dtype='float32'
)
decoder_targets_one_hot.shape

for j, d in enumerate(decoder_input_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[j, t, word] = 1
encoder_inputs_placeholder = Input(shape=(maxlen,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(maxlen,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history=model.fit([encoder_input_sequences_train,decoder_input_sequences_train],decoder_targets_one_hot[0:1144],epochs=20,batch_size=Batch_Size,validation_split=0.33,)
train_acc = model.evaluate([encoder_input_sequences_train, decoder_input_sequences_train],decoder_targets_one_hot[0:1144],verbose=0)
test_acc = model.evaluate([encoder_input_sequences_test, decoder_input_sequences_test],decoder_targets_one_hot[1144:1708],verbose=0)
print (train_acc, test_acc)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(' model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
from keras.utils import plot_model
plot_model(model, to_file='model_plot4.png', show_shapes=True, show_layer_names=True)
encoder_model = Model(encoder_inputs_placeholder, encoder_states)
decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)


idx2word_input = {v:k for k, v in word_index.items()}
idx2word_target = {v:k for k, v in word_index1.items()}
def decode_sequence(input_seq):
    states_value=encoder_model.predict(input_seq)
    target_seq=np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = word_index1['sos']
    eos = word_index1['eos']
    output_sentence = []
   
    for _ in range(maxlen):
        
        output_tokens,h,c=decoder_model.predict([target_seq]+states_value)
        idx = np.argmax(output_tokens[0, 0, :])
        
        
        if(idx==eos):
            break
        word=''
        
        if idx!=0:
            word=idx2word_target[idx]
            output_sentence.append(word)
        target_seq[0,0]=idx
        states_value=[h,c]
    return output_sentence
i =np.random.choice(len(encoder_input_sequences_test))

input_seq = encoder_input_sequences_test[i:i+1]
keyphrase_extraction = decode_sequence(input_seq)
print('-')
print('Input:', text[i])
print('Original:',label[i])
print('Response:', keyphrase_extraction)
        



    
   
encoder_input_sequences_train,encoder_input_sequences_test,decoder_input_sequences_train,decoder_input_sequences_test=train_test_split(encoder_input_sequences,decoder_input_sequences,test_size=0.33,random_state=42)        

nb_words=min(max_features,len(word_index))
#print(nb_words)
embedding_matrix=np.zeros((nb_words+1,embed_size))      #50000,embed size rakhlam


for embed_word,v in word_index.items():
    if v>=max_features:continue
    embedding_vector=vocab_vector.get(embed_word)
    
    
        #print(embedding_vector)
        
    #words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector 

embedding_layer=Embedding(num_words,embed_size,input_length=maxlen)
decoder_targets_one_hot = np.zeros((
        len(text),
        maxlen,
        num_words_output
        
    ),
    dtype='float32'
)
decoder_targets_one_hot.shape

for j, d in enumerate(decoder_input_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[j, t, word] = 1
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
history=model.fit([encoder_input_sequences_train,decoder_input_sequences_train],decoder_targets_one_hot[0:1144]
    ,validation_split=0.3,epochs=15,batch_size=64)
train_acc = model.evaluate([encoder_input_sequences_train, decoder_input_sequences_train],decoder_targets_one_hot[0:1144],
                           verbose=0)
print (train_acc)
test_acc = model.evaluate([encoder_input_sequences_test, decoder_input_sequences_test],decoder_targets_one_hot[1144:],
                          verbose=0)
print (test_acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(' model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
#decoder at test time
#encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs_placeholder, encoder_states)
#decoder set up
#below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
idx2word_input = {v:k for k, v in word_index.items()}
idx2word_target = {v:k for k, v in word_index1.items()}

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_word_index['sos']
    eos = output_word_index['eos']
    output_sentence = []

    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break
        word = ''
        
        if idx > 0:
            
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(set(output_sentence))
   
i =np.random.choice(len(encoder_input_sequences_test))

input_seq = encoder_input_sequences_test[i:i+1]
keyphrase_extraction = decode_sequence(input_seq)
print('-')
print('Input:', text[i])
print('Original:',label[i])
print('Response:', keyphrase_extraction)
        