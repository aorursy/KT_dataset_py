import collections

from collections import Counter



import helper

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential

from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional,LSTM

from keras.layers.embeddings import Embedding

from keras.optimizers import Adam

from keras.losses import sparse_categorical_crossentropy

from keras.callbacks import ModelCheckpoint



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



from sklearn.model_selection import train_test_split



from tabulate import tabulate



import gc
url = '../input/language-translation-englishfrench/eng_-french.csv'



data = pd.read_csv(url, header='infer')
#Total Records

print("Total Records: ", data.shape[0])
#Checking for Null/Missing Values

data.isna().sum()
#Renaming Columns

data = data.rename(columns={"English words/sentences":"Eng", "French words/sentences":"Frn" })
#Randomly Show a English > French sentence

x = np.random.randint(1, data.shape[0])

print("--- Random English - French Sentence --- \n"

      "English Sentence/Word: ", data.Eng[x], "\n"

      "French Sentence/Word: ", data.Frn[x]

     )
# Function for word count

def word_count (txt):

    return len(txt.split())
#Applying the Word Count Function to Eng & French Columns

data['Eng_Count'] = data['Eng'].apply(lambda x: word_count(x))

data['Frn_Count'] = data['Frn'].apply(lambda x: word_count(x))
print( '{} English Words'.format(data['Eng_Count'].sum()) ) 

print('{} French Words'.format(data['Frn_Count'].sum()) )

      
fig = make_subplots(rows=1, cols=2, subplot_titles=("English","French"))



fig.add_trace(

    go.Histogram(x=data['Eng_Count'],histfunc='sum',opacity =0.8,showlegend=False,text='Eng'), row=1,col=1)



fig.add_trace(

    go.Histogram(x=data['Frn_Count'],histfunc='sum', opacity =0.8,showlegend=False,text='Frn'), row=1,col=2)



fig.update_layout(height=600, width=800, title_text="Words Distribution")

fig.show()



#Tokenize Function

def tokenize(x):

    x_tk = Tokenizer(char_level = False)

    x_tk.fit_on_texts(x)

    return x_tk.texts_to_sequences(x), x_tk

    #return x_tk

#Padding Function

def pad(x, length=None):

    if length is None:

        length = max([len(sentence) for sentence in x])

    return pad_sequences(x, maxlen = length, padding = 'post')

    
#Tokenize English text & determine English Vocab Size 

eng_seq, eng_tok = tokenize(data['Eng'])

eng_vocab_size = len(eng_tok.word_index) + 1

print("Complete English Vocab Size: ",eng_vocab_size)



#Tokenize French text & determine French Vocab Size 

frn_seq, frn_tok = tokenize(data['Frn'])

frn_vocab_size = len(frn_tok.word_index) + 1

print("Complete French Vocab Size: ",frn_vocab_size)

#Sequence Length (Complete Dataset) 

eng_len = max([len(sentence) for sentence in eng_seq])

frn_len = max([len(sentence) for sentence in frn_seq])



print("English Sequence Length: ",eng_len,"\n",

      "French Sequence Length: ",frn_len)
# split data into train (90%) and test set (10%)

train_data, test_data = train_test_split(data, test_size=0.1, random_state = 0)
#Drop Columns

train_data = train_data.drop(columns=['Eng_Count', 'Frn_Count'],axis=1)

test_data = test_data.drop(columns=['Eng_Count', 'Frn_Count'],axis=1)



#Re-Index

train_data = train_data.reset_index(drop=True)

test_data = test_data.reset_index(drop=True)

# -- Tokenization --



# Training Data

train_X_seq, train_X_tok = tokenize(train_data['Eng'])

train_Y_seq, train_Y_tok = tokenize(train_data['Frn'])



train_eng_vocab = len(train_X_tok.word_index) + 1

train_frn_vocab = len(train_Y_tok.word_index) + 1



# Testing Data

test_X_seq, test_X_tok = tokenize(test_data['Eng'])

test_Y_seq, test_Y_tok = tokenize(test_data['Frn'])



test_eng_vocab = len(test_X_tok.word_index) + 1

test_frn_vocab = len(test_Y_tok.word_index) + 1





# -- Padding --



#Training Data

train_X_seq = pad(train_X_seq)

train_Y_seq = pad(train_Y_seq)



#Testing Data

test_X_seq = pad(test_X_seq)

test_Y_seq = pad(test_Y_seq)



#Tabulate the Vocab Size

tab_data = [["Train", train_eng_vocab, train_frn_vocab],["Test",test_eng_vocab,test_frn_vocab]]

print(tabulate(tab_data, headers=['Dataset','Eng Vocab Size','Frn Vocab Size'], tablefmt="pretty"))

# Define Model



def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps, btch_size):

    

    model = Sequential()

    model.add(Embedding(in_vocab, btch_size, input_length=in_timesteps, mask_zero=True))

    

    model.add(LSTM(btch_size))

    model.add(RepeatVector(out_timesteps))

    model.add(LSTM(btch_size, return_sequences=True))

    model.add(Dense(out_vocab, activation='softmax'))

    

    return model
# Compile Parameters

batch_size = 64   #batch size

lr = 1e-3          #learning rate



#Model

model = define_model(eng_vocab_size, frn_vocab_size, eng_len, frn_len, batch_size)



#Compile Model

model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(lr))

fn = 'model.h1.MT'

epoch = 2

val_split = 0.1



#Checkpoint

checkpoint = ModelCheckpoint(fn, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



#Train

history = model.fit(train_X_seq, train_Y_seq,

                    epochs=epoch, batch_size=batch_size, validation_split = val_split, callbacks=[checkpoint], 

                    verbose=1)



plt.rcParams["figure.figsize"] = (10,8)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['train','validation'])

plt.title("Train vs Validation - Loss", fontsize=15)

plt.show()
#Making Prediction

predictions = model.predict(test_X_seq[1:6])[0]
def to_text(logits, tokenizer):



    index_to_words = {id: word for word, id in tokenizer.word_index.items()}

    index_to_words[0] = ''

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
print(to_text(predictions, frn_tok))