import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sentencepiece as sp

import time

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from numpy.random import shuffle



init_notebook_mode(connected=True)
def read_text(filename): 

        # open the file 

        file = open(filename, mode='rt', encoding='utf-8') 

        

        # read all text 

        text = file.read() 

        file.close() 

        return text
file_path = "/kaggle/input/en-dutch-pairs/nld.txt"
start_time = time.time()

train_df = pd.read_csv(file_path, sep='\t', lineterminator='\n', names=["EN","NL"])

print(f"File loading time: {time.time()-start_time}")

print(train_df.shape)
train_df.head()
def write_trainer_file(col, filename):

    texts = list(col.values)

    with open(filename, 'w',encoding='utf-8') as f:

        for text in texts:

            f.write(text + "\n")
en_sp_trainer = "en_spm.txt"

write_trainer_file(train_df["EN"], en_sp_trainer)
nl_sp_trainer = "nl_spm.txt"

write_trainer_file(train_df["NL"], nl_sp_trainer)
def createSPModel(trainer_file, model_prefix, vocab_size, sp):

    spm_train_param = f"--input={trainer_file} --model_prefix={model_prefix} --vocab_size={vocab_size}"

    sp.SentencePieceTrainer.Train(spm_train_param)

    lang_sp = sp.SentencePieceProcessor()

    lang_sp.Load(f"{model_prefix}.model")

    return lang_sp

    
en_sp = createSPModel(en_sp_trainer, "en_sp", 7150, sp)
print(en_sp.EncodeAsPieces("This is a test."))

print(en_sp.EncodeAsIds("This is a test."))

print(en_sp.DecodeIds(en_sp.EncodeAsIds("This is a test.")))
nl_sp = createSPModel(nl_sp_trainer, "nl_sp", 9600, sp)
nl_sp.EncodeAsPieces("Ik wil graag een fles water.")
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding, RepeatVector

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras import optimizers

from sklearn.model_selection import train_test_split
def encode_sentence(df, lang, spm):

    lang_pieces = []

    lang_lens = []

    for index, row in df.iterrows():

        lang_piece = spm.EncodeAsIds(row[lang])

        lang_pieces.append(lang_piece)

        lang_lens.append(len(lang_piece)) 

    df[f"{lang}_pieces"] = lang_pieces

    df[f"{lang}_len"] = lang_lens
start_time = time.time()

encode_sentence(train_df, "EN", en_sp)

encode_sentence(train_df, "NL", nl_sp)

print(f"Encoding time: {time.time()-start_time} sec")

train_df.tail()
def plotLangLen(lang1, lang2):

    trace1 = go.Histogram(

        x=train_df[f"{lang1}_len"].values,

        opacity=0.75,

        name = f"Length of {lang1} sentences",

        marker=dict(color='rgba(171, 50, 96, 0.6)'))

    trace2 = go.Histogram(

        x=train_df[f"{lang2}_len"].values,

        opacity=0.75,

        name = f"Length of {lang2} sentences",

        marker=dict(color='rgba(12, 50, 196, 0.6)'))



    data = [trace1, trace2]

    layout = go.Layout(barmode='overlay',

                       title=f"Lengths of {lang1} and {lang2} sentences",

                       xaxis=dict(title='Length'),

                       yaxis=dict( title='Count'),

    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, config={'showLink': True})
plotLangLen("EN", "NL")
en_vocab_size = en_sp.get_piece_size()

nl_vocab_size = nl_sp.get_piece_size()

print(f"EN vocab size: {en_vocab_size}")

print(f"NL vocab size: {nl_vocab_size}")
print(en_sp.piece_to_id('__MUST_BE_UNKNOWN__'))

print(en_sp.id_to_piece(0))
en_max_length = train_df["EN_len"].max()

nl_max_length = train_df["NL_len"].max()

#30 for faster processing time

en_max_length=30

nl_max_length=en_max_length

print(en_max_length)

print(nl_max_length)
#use post padding to fill up short sentence with 0

en_padded_seq = pad_sequences(train_df["EN_pieces"].tolist(), maxlen=en_max_length, padding='post')

nl_padded_seq = pad_sequences(train_df["NL_pieces"].tolist(), maxlen=nl_max_length, padding='post')
#pick a sample

en_padded_seq[2]
train_seq_df = pd.DataFrame( {'en_seq':en_padded_seq.tolist(), 'nl_seq':nl_padded_seq.tolist()})
train_seq_df.tail()
def define_model(input_vocab,output_vocab, input_length,output_length,output_dim):

      model = Sequential()

      #mark_zero , set 0 as special character reserved for unknown words  

      model.add(Embedding(input_vocab, output_dim, input_length=input_length, mask_zero=True))

      model.add(LSTM(output_dim))

      #repeat the input (n) times

#return_sequences=True returns all the outputs the encoder observed in the past, while RepeatVector repeats the very last output of the encoder.    

      model.add(RepeatVector(output_length))

    #return the full sequences

      model.add(LSTM(output_dim, return_sequences=True))

      #model.add(TimeDistributed(Dense(output_vocab, activation='softmax')))

      

      model.add(Dense(output_vocab, activation='softmax'))

      return model
train_seq_df.shape
train, test = train_test_split(train_seq_df, test_size=0.1, random_state = 3)
'''

#K-fold didn't help much in this case



kfold = KFold(n_splits=3)

k_preds = []

for i, (train_set, val_set ) in enumerate(kfold.split(train)):

  print(f"Running fold:{i}") 

  train_set_X = np.asarray(train.iloc[ train_set, 1].tolist()) 

  train_set_Y = np.asarray(train.iloc[ train_set, 0].tolist())



  val_set_X = np.asarray(train.iloc[ val_set, 1].tolist()) 

  val_set_Y = np.asarray(train.iloc[ val_set, 0].tolist()) 



  k_model = None 

  k_model = define_model(ja_vocab_size, en_vocab_size, ja_max_length, en_max_length, 1024) 

  k_model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy')

  k_filename = f"nmt_model_{i}"

  k_checkpoint = ModelCheckpoint(k_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

  

  # train model

#  k_history = k_model.fit(train_set_X, train_set_Y.reshape(train_set_Y.shape[0], train_set_Y.shape[1], 1),

#                    epochs=10, batch_size=64, validationdata = (val_set_X, val_set_Y.reshape(val_set_Y.shape[0], val_set_Y.shape[1], 1)), callbacks=[checkpoint], 

#                    verbose=1)

  k_history = k_model.fit(train_set_X, train_set_Y.reshape(train_set_Y.shape[0], train_set_Y.shape[1], 1),

                    epochs=10, batch_size=64, callbacks=[k_checkpoint],validation_split = 0.2,

                    verbose=1)

  k_model = load_model(f"nmt_model_{i}")

  k_preds.append(model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1]))))



#  model = loadmodel(filepath) 

#  pred += model.predict(Xtest, batchsize = batchsizes, verbose = 1)

#  preds = pred/n_folds

#for train_set, val_set in kfold.split(train):    

#   trainX = np.asarray(train_seq_df.loc[ train, "ja_seq" ].tolist()) 

#   trainY = np.asarray(train_seq_df.loc[ train, "en_seq" ].tolist())

    

#   testX = np.asarray(train_seq_df.loc[ test, "ja_seq" ].tolist()) 

#   testY = np.asarray(train_seq_df.loc[ test, "en_seq" ].tolist()) 

'''
trainX = np.asarray(train["nl_seq"].tolist())

trainY = np.asarray(train["en_seq"].tolist())



testX = np.asarray(test["nl_seq"].tolist())

testY = np.asarray(test["en_seq"].tolist())
model = define_model(nl_vocab_size, en_vocab_size, nl_max_length, en_max_length, 1024)
#RMSprop is recommended for RNN, sparse_categorical_crossentropy for densed target output as integers

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy')
def encode_output(sequences, vocab_size):

   ylist = list()

   for sequence in sequences:

    encoded = to_categorical(sequence, num_classes=vocab_size)

    ylist.append(encoded)

    y = array(ylist)

    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)

    return y
filename = 'nmt_model'

checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



# train model

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),

                    epochs=15, batch_size=64, validation_split = 0.1,callbacks=[checkpoint], 

                    verbose=1)

trace1 = go.Scatter(

    y=history.history['loss'],

    name = "Training Loss",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Scatter(

    y=history.history['val_loss'],

    name = "Validation Loss",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(title='Loss and Val_Loss in 15 Epochs',

                   xaxis=dict(title='Epoch'),

                   yaxis=dict( title='Loss'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, config={'showLink': True})
model = load_model('nmt_model')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
def get_word(ids, tokenizer):

    return tokenizer.DecodeIds(list(filter(lambda a: a != 0, ids.tolist())))

import random
y_indexs = random.sample(range(len(testY)),  30)



for y_index in y_indexs: 

  print(f"--- Index: {y_index}")

  print(f"NL: {get_word(testX[y_index], nl_sp)}") 

  print(f"EN: {get_word(testY[y_index], en_sp)}") 

  print(f"MT: {get_word(preds[y_index], en_sp)}")

 # print(get_word(k_preds[0][y_index], en_sp))

 # print(get_word(k_preds[1][y_index], en_sp))

 # print(get_word(k_preds[2][y_index], en_sp))

test_ids = []

test_nls = []

test_ens = []

test_mts = []

for y_index in range(len(testY)): 

  test_ids.append(y_index)

  test_nls.append(get_word(testX[y_index], nl_sp))

  test_ens.append(get_word(testY[y_index], en_sp))

  test_mts.append(get_word(preds[y_index], en_sp))



predict_df = pd.DataFrame( {'id':test_ids, 'NL':test_nls, 'EN':test_ens, 'MT':test_mts})
pd.set_option('display.max_colwidth', 80)

predict_df.sample(10)