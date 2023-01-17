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
import numpy as np
import pandas as pd
import string
import pickle
import operator
import matplotlib.pyplot as plt
%matplotlib inline
import codecs
import gc
with codecs.open('/kaggle/input/movie-dialog-corpus/movie_lines.tsv','rb',encoding='utf-8',errors='ignore') as f:
    lines=f.read().split('\n')
    
conversations=[]
for line in lines:
    data=line.split('\t')
    conversations.append(data)
    
conversations[:11]
chats = {}
for tokens in conversations:
    if len(tokens) > 4:
        idx_L=tokens[0].find('L')
        if idx_L !=-1:
            idx=tokens[0][idx_L+1:]
            chat = tokens[4]
            chat=chat[:-2]
            chats[int(idx)] = chat
sorted_chats=sorted(chats.items(),key=lambda x:x[0])
sorted_chats[:5]
conves_dict = {}
counter = 1
conves_ids = []
for i in range(1, len(sorted_chats)+1):
    if i < len(sorted_chats):
        if (sorted_chats[i][0] - sorted_chats[i-1][0]) == 1:
            # 1つ前の会話の頭の文字がないのを確認
            if sorted_chats[i-1][1] not in conves_ids:
                conves_ids.append(sorted_chats[i-1][1])
            conves_ids.append(sorted_chats[i][1])
        elif (sorted_chats[i][0] - sorted_chats[i-1][0]) > 1:            
            conves_dict[counter] = conves_ids
            conves_ids = []
        counter += 1
    else:
        pass
conves_dict[3]
context_and_target=[]
for conves in conves_dict.values():
    if len(conves) % 2 != 0:
        conves = conves[:-1]
    for i in range(0, len(conves), 2):
        context_and_target.append((conves[i], conves[i+1]))
context_and_target[:5]
context, target = zip(*context_and_target)
context = list(context)
target = list(target)
context[:5]
target[:5]
import re
def clean_text(text):    

    text = text.lower()    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text
tidy_target = []
for conve in target:
    text = clean_text(conve)
    tidy_target.append(text)
tidy_target[:20]
tidy_context = []
for conve in context:
    text = clean_text(conve)
    tidy_context.append(text)
tidy_context[:20]
bos = "<BOS> "
eos = " <EOS>"
final_target = [bos + conve + eos for conve in tidy_target] 
encoder_inputs = tidy_context
decoder_inputs = final_target
encoder_text = []
for line in encoder_inputs:
    data = line.split("\n")[0]
    encoder_text.append(data)
len(encoder_text)
encoder_text[:5]
decoder_text = []
for line in decoder_inputs:
    data = line.split("\n")[0]
    decoder_text.append(data)
len(decoder_text)
decoder_text[:5]
full_text=encoder_text+decoder_text
from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 15000
tokenizer = Tokenizer(num_words=VOCAB_SIZE,oov_token='<OOV>')
tokenizer.fit_on_texts(full_text)
word_index = tokenizer.word_index
print(len(word_index))
word_index[bos]=len(word_index)+1
word_index[eos]=len(word_index)+1
index2word = {}
for k, v in word_index.items():
    if v < VOCAB_SIZE:
        index2word[v] = k
    if v > VOCAB_SIZE:
        continue
len(index2word)
word2index = {}
for k, v in index2word.items():
    word2index[v] = k
len(word2index)
encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

encoder_sequences[:5]

for seqs in encoder_sequences:
    for seq in seqs:
        if seq > VOCAB_SIZE:
            print(seq)
            break
VOCAB_SIZE = len(index2word) + 1
VOCAB_SIZE
decoder_sequences[:5]
MAX_LEN = 20
from keras.preprocessing.sequence import pad_sequences
encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
import numpy as np

num_samples = len(encoder_sequences)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
for i, seqs in enumerate(decoder_input_data):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1
decoder_output_data.shape
decoder_input_data[0]
del encoder_sequences
del decoder_sequences
del full_text
del encoder_text
del decoder_text
del tidy_context
del final_target
del tidy_target
del target
del context
del context_and_target
del conves_dict
gc.collect()
embeddings_index = {}
with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")
embedding_dimention = 100
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(embedding_dimention, word_index=word2index)

from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model
embed_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dimention, trainable=True,)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])
HIDDEN_DIM=300
    
encoder_inputs = Input(shape=(None, ), dtype='int32',)
encoder_embedding = embed_layer(encoder_inputs)
encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
encoder_states=[state_h,state_c]

decoder_inputs = Input(shape=(None, ), dtype='int32',)
decoder_embedding = embed_layer(decoder_inputs)
decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)

decoder_state_input_h=Input(shape=(HIDDEN_DIM,))
decoder_state_input_c=Input(shape=(HIDDEN_DIM,))

decoder_state_inputs=[decoder_state_input_h,decoder_state_input_c]
new_decoder_outputs,new_state_h,new_state_c=decoder_LSTM(decoder_embedding,initial_state=decoder_state_inputs)

decoder_states=[new_state_h,new_state_c]
new_decoder_outputs=TimeDistributed(Dense(VOCAB_SIZE,activation='softmax'))(new_decoder_outputs)

outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)    
 
    
  
model.summary()
from keras.utils import plot_model
plot_model(model)
plot_model(encoder_model)
plot_model(decoder_model)
model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
BATCH_SIZE = 128
EPOCHS = 1
encoder_input_data.shape
history = model.fit([encoder_input_data, decoder_input_data], 
                     decoder_output_data, 
                     epochs=EPOCHS, validation_split=0.2,                    
                     batch_size=BATCH_SIZE)
model.save('my_model')
print(model.input)
print()

print(model.layers[0])
print(model.layers[1])
print(model.layers[2])
print(model.layers[3])
print(model.layers[4])
print(model.layers[5])
def encoder_decoder_model(model):
    encoder_inputs=model.input[0]
    embedding_layer=model.layers[2]
    
    encoder_embedding=embedding_layer(encoder_inputs)
    encoder_lstm=model.layers[3]
    encoder_output,state_h,state_c=encoder_lstm(encoder_embedding)
    encoder_states=[state_h,state_c]
    
    decoder_inputs=model.input[1]
    decoder_embedding=embedding_layer(decoder_inputs)
    decoder_lstm=model.layers[4]
    decoder_outputs,_,_=decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    
    decoder_state_input_h=Input(shape=(300,))
    decoder_state_input_c=Input(shape=(300,))
    
    decoder_state_inputs=[decoder_state_input_h,decoder_state_input_c]
    new_decoder_outputs,new_state_h,new_state_c=decoder_lstm(decoder_embedding,initial_state=decoder_state_inputs)
    decoder_states=[new_state_h,new_state_c]
    new_decoder_outputs=model.layers[5](new_decoder_outputs)
    

    encoder_model=Model(encoder_inputs,encoder_states)
    decoder_model=Model([decoder_inputs]+decoder_state_inputs,[new_decoder_outputs]+decoder_states)
    
    return encoder_model,decoder_model
    
    
    
    
    
    
encoder_model,decoder_model=encoder_decoder_model(model)
plot_model(encoder_model)
plot_model(decoder_model)
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model accuracy')
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model loss')
model.save('same_model.h5')
def str_to_tokens(sentence:str):
    
    words=sentence.lower().split()
    tokens_list=list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return pad_sequences([tokens_list],maxlen=20,padding='post')
sentences=["hey how are you",
           "we should go out and play",
           "voilence is bad we should never do it",
           "sharing is caring",
           "i am going to the football game, want to come?"]


for i in range(5):
    states_values=encoder_model.predict(str_to_tokens(sentences[i]))
    empty_target_seq=np.zeros((1,1))
    empty_target_seq[0,0]=tokenizer.word_index[bos]
    stop_condition=False
    decoded_translation=""
    
    while not stop_condition:
        dec_outputs,h,c=decoder_model.predict([empty_target_seq]+states_values)
        sampled_word_index=np.argmax(dec_outputs[0,-1,:])
        
        
        sampled_word=None
        for word,index in tokenizer.word_index.items():
            if sampled_word_index==index:
                decoded_translation+=' {}'.format(word)
                sampled_word=word
                
        if sampled_word ==eos or len(decoded_translation.split())>20:
            stop_condition=True
        
        
        empty_target_seq=np.zeros((1,1))
        
        empty_target_seq[0,0]=sampled_word_index
        #print("empty target sequence",empty_target_seq)
        states_values=[h,c]
        
    print(decoded_translation)






















