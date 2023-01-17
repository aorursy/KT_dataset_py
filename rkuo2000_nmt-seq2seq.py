import os 
print(os.listdir('../input/textdata'))
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import matplotlib.pyplot as plt
filename = '../input/textdata/deu-eng.txt' # English-to-Deutsche

f = open(filename, mode='rt', encoding='utf-8')
data = f.read()
f.close()
# split a text into sentences
def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents
deu_eng = to_lines(data)
deu_eng = array(deu_eng)    #total = 150,000 sentences
deu_eng = deu_eng[:50000,:] #train =  50,000 sentences 

# Remove punctuation
deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]]
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]]
# convert text to lowercase
for i in range(len(deu_eng)):
    deu_eng[i,0] = deu_eng[i,0].lower()
    deu_eng[i,1] = deu_eng[i,1].lower()
# empty lists
eng_l = []
deu_l = []

# populate the lists with sentence lengths
for i in deu_eng[:,0]:
    eng_l.append(len(i.split()))

for i in deu_eng[:,1]:
    deu_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})

length_df.hist(bins = 30)
plt.show()
# function to build a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
# prepare english tokenizer
eng_tokenizer = tokenization(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)
# prepare Deutch tokenizer
deu_tokenizer = tokenization(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = 8
print('Deutch Vocabulary Size: %d' % deu_vocab_size)
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)               # integer encode sequences    
    seq = pad_sequences(seq, maxlen=length, padding='post') # pad sequences with 0 values
    return seq
# split data into train and test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(deu_eng, test_size=0.1, random_state = 12)

# prepare training data
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
in_vocab     = deu_vocab_size
out_vocab    = eng_vocab_size
in_timesteps = deu_length
out_timesteps= eng_length
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
# Build Model
model = Sequential()
model.add(Embedding(in_vocab, 512, input_length=in_timesteps, mask_zero=True))
model.add(LSTM(512))
model.add(RepeatVector(out_timesteps))
model.add(LSTM(512, return_sequences=True))
model.add(Dense(out_vocab, activation='softmax'))

model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
keras.utils.plot_model(model = model , to_file = 'Seq2Seq.png')
#model_path = "./seq2seq_chkpt.h5"
#checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
num_epochs = 50
batch_size = 512
# Train Model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
        epochs=num_epochs, 
        batch_size=batch_size, 
        validation_split = 0.1,
#        callbacks=[checkpoint], 
        verbose=1)
# Save Model
model.save("./seq2seq_deu-eng.h5")
# Plot Training History
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()
# Test Model
preds = model.predict_classes(testX)
def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None
# convert predictions into text	  
preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t) 
    preds_text.append(' '.join(temp))
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
print(pred_df.sample(15)) # print 15 rows randomly