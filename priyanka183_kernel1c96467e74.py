import os
import numpy as np
import pandas as pd
from numpy import array
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.models import Model
from keras.layers import TimeDistributed,Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
bangla_fullstop = re.compile(u"\u0964",re.UNICODE)
punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
print(os.listdir("../input"))

Batch_Size=20
max_features=50000
maxlen=457
embed_size=300
input_sentences=[]
#n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.

EMBEDDING_FILE='../input/bengali-word-embedding/bengali-word-embedding.txt'
# put words as dict indexes and vectors as words values
from time import time
t = time()
vocab_vector = {} 
with open(EMBEDDING_FILE,encoding='utf8') as f:  
    for line in f:
        values = line.rstrip().rsplit(' ')
        word_values = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        vocab_vector[word_values] = coefs
f.close()   
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

        
    


data = pd.read_csv('../input/filename-2712/filename.csv')
text1=data['article']
print(text1[1])
data.keys()


def remove_punctuation(text):
   
    text = whitespace.sub(" ",text).strip()
    text = re.sub(punctSeq, " ", text)
    text = re.sub("৷", " ",text)
    text = re.sub(punc, " ", text)
    return text
data['article']=data['article'].apply(lambda x: remove_punctuation(x))
stop_words=["অতএব","অথচ","অথবা","অনুযায়ী","অনেক","অনেকে","অনেকেই","অন্তত","অন্য","অবধি","অবশ্য","অর্থাত","আই","আগামী","আগে","আগেই","আছে","আজ","আদ্যভাগে","আপনার","আপনি","আবার","আমরা","আমাকে","আমাদের","আমার","আমি","আর","আরও","ই","ইত্যাদি","ইহা","উচিত","উত্তর","উনি","উপর","উপরে","এ","এঁদের","এঁরা","এই","একই","একটি","একবার","একে","এক্","এখন","এখনও","এখানে","এখানেই","এটা","এটাই","এটি","এত","এতটাই","এতে","এদের","এব","এবং","এবার","এমন","এমনকী","এমনি","এর","এরা","এল","এস","এসে","ঐ","ও","ওঁদের","ওঁর","ওঁরা","ওই","ওকে","ওখানে","ওদের","ওর","ওরা","কখনও","কত","কবে","কমনে","কয়েক","কয়েকটি","করছে","করছেন","করতে","করবে","করবেন","করলে","করলেন","করা","করাই","করায়","করার","করি","করিতে","করিয়া","করিয়ে","করে","করেই","করেছিলেন","করেছে","করেছেন","করেন","কাউকে","কাছ","কাছে","কাজ","কাজে","কারও","কারণ","কি","কিংবা","কিছু","কিছুই","কিন্তু","কী","কে","কেউ","কেউই","কেখা","কেন","কোটি","কোন","কোনও","কোনো","ক্ষেত্রে","কয়েক","খুব","গিয়ে","গিয়েছে","গিয়ে","গুলি","গেছে","গেল","গেলে","গোটা","চলে","চান","চায়","চার","চালু","চেয়ে","চেষ্টা","ছাড়া","ছাড়াও","ছিল","ছিলেন","জন","জনকে","জনের","জন্য","জন্যওজে","জানতে","জানা","জানানো","জানায়","জানিয়ে","জানিয়েছে","জে","জ্নজন","টি","ঠিক","তখন","তত","তথা","তবু","তবে","তা","তাঁকে","তাঁদের","তাঁর","তাঁরা","তাঁাহারা","তাই","তাও","তাকে","তাতে","তাদের","তার","তারপর","তারা","তারৈ","তাহলে","তাহা","তাহাতে","তাহার","তিনঐ","তিনি","তিনিও","তুমি","তুলে","তেমন","তো","তোমার","থাকবে","থাকবেন","থাকা","থাকায়","থাকে","থাকেন","থেকে","থেকেই","থেকেও","দিকে","দিতে","দিন","দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুই","দুটি","দুটো","দেওয়া","দেওয়ার","দেওয়া","দেখতে","দেখা","দেখে","দেন","দেয়","দ্বারা","ধরা","ধরে","ধামার","নতুন","নয়","না","নাই","নাকি","নাগাদ","নানা","নিজে","নিজেই","নিজেদের","নিজের","নিতে","নিয়ে","নিয়ে","নেই","নেওয়া","নেওয়ার","নেওয়া","নয়","পক্ষে","পর","পরে","পরেই","পরেও","পর্যন্ত","পাওয়া","পাচ","পারি","পারে","পারেন","পি","পেয়ে","পেয়্র্","প্রতি","প্রথম","প্রভৃতি","প্রযন্ত","প্রাথমিক","প্রায়","প্রায়","ফলে","ফিরে","ফের","বক্তব্য","বদলে","বন","বরং","বলতে","বলল","বললেন","বলা","বলে","বলেছেন","বলেন","বসে","বহু","বা","বাদে","বার","বি","বিনা","বিভিন্ন","বিশেষ","বিষয়টি","বেশ","বেশি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মতো","মতোই","মধ্যভাগে","মধ্যে","মধ্যেই","মধ্যেও","মনে","মাত্র","মাধ্যমে","মোট","মোটেই","যখন","যত","যতটা","যথেষ্ট","যদি","যদিও","যা","যাঁর","যাঁরা","যাওয়া","যাওয়ার","যাওয়া","যাকে","যাচ্ছে","যাতে","যাদের","যান","যাবে","যায়","যার","যারা","যিনি","যে","যেখানে","যেতে","যেন","যেমন","র","রকম","রয়েছে","রাখা","রেখে","লক্ষ","শুধু","শুরু","সঙ্গে","সঙ্গেও","সব","সবার","সমস্ত","সম্প্রতি","সহ","সহিত","সাধারণ","সামনে","সি","সুতরাং","সে","সেই","সেখান","সেখানে","সেটা","সেটাই","সেটাও","সেটি","স্পষ্ট","স্বয়ং","হইতে","হইবে","হইয়া","হওয়া","হওয়ায়","হওয়ার","হচ্ছে","হত","হতে","হতেই","হন","হবে","হবেন","হয়","হয়তো","হয়নি","হয়ে","হয়েই","হয়েছিল","হয়েছে","হয়েছেন","হল","হলে","হলেই","হলেও","হলো","হাজার","হিসাবে","হৈলে","হোক","হয়"]
def remove_stopwords(text,remove_stop_words=True):
    
    if remove_stop_words:
        
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    return text  
data['article']=data['article'].apply(lambda x: remove_stopwords(x))

               
text2 = data['article']
print(text2[1])
    
#data['manualKeyPhrases']=data['manualKeyPhrases'].apply(lambda x: x).astype(str)
#data['manualKeyPhrases']=data['manualKeyPhrases'].apply(lambda x: remove_punctuation(x))
#data['manualKeyPhrases']=data['manualKeyPhrases'].apply(lambda x : 'sos '+ x + ' eos')
#label = data['manualKeyPhrases']
#print(label[1])
outputs=[]
outputs1=[]

  
for l in text2:
    s = l + 'eos'
    s1='sos' + l
    outputs.append(s)
    outputs1.append(s1)

max_features=1000
input_tokenizer = Tokenizer(num_words=max_features)
input_tokenizer.fit_on_texts(text1)
input_sequence = input_tokenizer.texts_to_sequences(text1)
input_word_index = input_tokenizer.word_index
num_words=len(input_word_index)+1
print('Found %s unique tokens.' %num_words)
max_input_len = max(len(sen) for sen in input_sequence)
print("Length of longest sentence in input: %g" % max_input_len)
print(input_word_index["বাংলাদেশ"])
encoder_input_sequences= pad_sequences(input_sequence, maxlen=max_input_len, padding='pre', truncating='post')
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)

output_tokenizer  = Tokenizer(num_words=1000) 
output_tokenizer.fit_on_texts(outputs+outputs1)
output_sequence1 = output_tokenizer .texts_to_sequences(outputs)
output_sequence2 = output_tokenizer .texts_to_sequences(outputs1)
output_word_index = output_tokenizer.word_index
y_voc  =   output_tokenizer.num_words + 1
print('Found %s unique tokens.' %y_voc)
max_output_len = max(len(sen) for sen in output_sequence1)


print("Length of longest sentence in output: %g" % max_output_len)

decoder_input_sequences=pad_sequences(output_sequence2,maxlen=max_output_len,padding='post',truncating='post')    
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)

nb_words=min(max_features,len(input_word_index))
print(nb_words)
embedding_matrix=np.zeros((nb_words+1,embed_size))      #50000,embed size rakhlam


for embed_word,v in input_word_index.items():
    if v>=max_features:continue
    embedding_vector=vocab_vector.get(embed_word)
        #print(embedding_vector)
    #words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector 
encoder_input_sequences_train,encoder_input_sequences_test,decoder_input_sequences_train,decoder_input_sequences_test=train_test_split(encoder_input_sequences,decoder_input_sequences,test_size=0.33,random_state=42)        
print(len(encoder_input_sequences_train))
embedding_layer=Embedding(max_features+1,embed_size,weights=[embedding_matrix],input_length=max_input_len)
decoder_targets_one_hot = np.zeros((
        len(text),
        max_output_len,
        y_voc
    ),
    dtype='float32'
)
decoder_targets_one_hot.shape
for j, d in enumerate(decoder_input_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[j, t, word] = 1 
        
LSTM_NODES=128
dropout_rate=0.2    #rate of the dropout layers
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True, return_sequences=True, dropout=0.4,recurrent_dropout=0.4)

encoder_output1, h, c = encoder(x)
encoder_states = [h, c]

#encoder lstm 2
encoder_lstm2 = LSTM(LSTM_NODES,return_state=True,return_sequences=True, dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h, state_c = encoder_lstm2(encoder_output1)
#encoder lstm 3
encoder_lstm3=LSTM(LSTM_NODES, return_state=True, dropout=0.2,recurrent_dropout=0.2)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

decoder_inputs_placeholder = Input(shape=(max_output_len,))

decoder_embedding = Embedding(y_voc, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(y_voc, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
       
        
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
history=model.fit([encoder_input_sequences_train,decoder_input_sequences_train],decoder_targets_one_hot[0:1814]
    ,validation_split=0.33,epochs=10,batch_size=64)
train_acc = model.evaluate([encoder_input_sequences_train, decoder_input_sequences_train],decoder_targets_one_hot[0:1814],
                           verbose=0)
print (train_acc)
test_acc = model.evaluate([encoder_input_sequences_test, decoder_input_sequences_test],decoder_targets_one_hot[1814:],
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
reverse_target_word_index=output_tokenizer.index_word
reverse_source_word_index=input_tokenizer.index_word
target_word_index=output_tokenizer.word_index

def seq2key(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sos']) and i!=target_word_index['eos']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString
idx2word_input = {v:k for k, v in input_word_index.items()}
idx2word_target = {v:k for k, v in output_word_index.items()}

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    #target_seq[0,0]= output_word_index['sos']
    #eos = output_word_index['eos']
    output_sentence = []

    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if 'eos' == idx:
            break
        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(set(output_sentence))
       
i = np.random.choice(len(encoder_input_sequences_test))
print(i)
input_seq = encoder_input_sequences_test[i:i+1]
translation = translate_sentence(input_seq)
print('-')
print('Input:', text1[i])
print('Input:', seq2key(decoder_input_sequences_test[i]))
print('Original:', text2[i])
print('Response:', translation)