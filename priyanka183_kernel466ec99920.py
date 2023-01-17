import os # for manipulates files and subdirectories
import json # handle json files
import csv
import codecs
import nltk
from nltk.tokenize import word_tokenize
import re
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Input,LSTM,GRU,Dense,Embedding
from keras.models import Model
import matplotlib.pyplot as plt

whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
bangla_fullstop = re.compile(u"\u0964",re.UNICODE)
punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"

import string
max_features=26637

embed_size=300
#maxlen=200
LSTM_NODES=128
Batch_Size=32

#print(os.listdir("../input"))
#json_folder_path = os.path.join("../input/A 500")
json_folder_path = os.path.join("../input/json-data-a500/A 500")
# In order to get the list of all files that ends with ".json"
# we will get list of all files, and take only the ones that ends with "json"
json_files = [ x for x in os.listdir(json_folder_path) if x.endswith("json") ]
data=list() 
for json_file in json_files:
    json_file_path = os.path.join(json_folder_path, json_file)
    with open(json_file_path,"r",encoding="utf-8") as f:
        data.append(json.load(f))
        dataframe=pd.DataFrame(data)
dataframe.to_csv('filename.csv',encoding='utf-8',index= True)


def remove_punctuation(text):
   
    text = whitespace.sub(" ",text).strip()
    text = re.sub(punctSeq, " ", text)
    text = re.sub("৷", " ",text)
    text = re.sub(punc, " ", text)
    return text
dataframe['article']=dataframe['article'].apply(lambda x: remove_punctuation(x))
stop_words=["অতএব","অথচ","অথবা","অনুযায়ী","অনেক","অনেকে","অনেকেই","অন্তত","অন্য","অবধি","অবশ্য","অর্থাত","আই","আগামী","আগে","আগেই","আছে","আজ","আদ্যভাগে","আপনার","আপনি","আবার","আমরা","আমাকে","আমাদের","আমার","আমি","আর","আরও","ই","ইত্যাদি","ইহা","উচিত","উত্তর","উনি","উপর","উপরে","এ","এঁদের","এঁরা","এই","একই","একটি","একবার","একে","এক্","এখন","এখনও","এখানে","এখানেই","এটা","এটাই","এটি","এত","এতটাই","এতে","এদের","এব","এবং","এবার","এমন","এমনকী","এমনি","এর","এরা","এল","এস","এসে","ঐ","ও","ওঁদের","ওঁর","ওঁরা","ওই","ওকে","ওখানে","ওদের","ওর","ওরা","কখনও","কত","কবে","কমনে","কয়েক","কয়েকটি","করছে","করছেন","করতে","করবে","করবেন","করলে","করলেন","করা","করাই","করায়","করার","করি","করিতে","করিয়া","করিয়ে","করে","করেই","করেছিলেন","করেছে","করেছেন","করেন","কাউকে","কাছ","কাছে","কাজ","কাজে","কারও","কারণ","কি","কিংবা","কিছু","কিছুই","কিন্তু","কী","কে","কেউ","কেউই","কেখা","কেন","কোটি","কোন","কোনও","কোনো","ক্ষেত্রে","কয়েক","খুব","গিয়ে","গিয়েছে","গিয়ে","গুলি","গেছে","গেল","গেলে","গোটা","চলে","চান","চায়","চার","চালু","চেয়ে","চেষ্টা","ছাড়া","ছাড়াও","ছিল","ছিলেন","জন","জনকে","জনের","জন্য","জন্যওজে","জানতে","জানা","জানানো","জানায়","জানিয়ে","জানিয়েছে","জে","জ্নজন","টি","ঠিক","তখন","তত","তথা","তবু","তবে","তা","তাঁকে","তাঁদের","তাঁর","তাঁরা","তাঁাহারা","তাই","তাও","তাকে","তাতে","তাদের","তার","তারপর","তারা","তারৈ","তাহলে","তাহা","তাহাতে","তাহার","তিনঐ","তিনি","তিনিও","তুমি","তুলে","তেমন","তো","তোমার","থাকবে","থাকবেন","থাকা","থাকায়","থাকে","থাকেন","থেকে","থেকেই","থেকেও","দিকে","দিতে","দিন","দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুই","দুটি","দুটো","দেওয়া","দেওয়ার","দেওয়া","দেখতে","দেখা","দেখে","দেন","দেয়","দ্বারা","ধরা","ধরে","ধামার","নতুন","নয়","না","নাই","নাকি","নাগাদ","নানা","নিজে","নিজেই","নিজেদের","নিজের","নিতে","নিয়ে","নিয়ে","নেই","নেওয়া","নেওয়ার","নেওয়া","নয়","পক্ষে","পর","পরে","পরেই","পরেও","পর্যন্ত","পাওয়া","পাচ","পারি","পারে","পারেন","পি","পেয়ে","পেয়্র্","প্রতি","প্রথম","প্রভৃতি","প্রযন্ত","প্রাথমিক","প্রায়","প্রায়","ফলে","ফিরে","ফের","বক্তব্য","বদলে","বন","বরং","বলতে","বলল","বললেন","বলা","বলে","বলেছেন","বলেন","বসে","বহু","বা","বাদে","বার","বি","বিনা","বিভিন্ন","বিশেষ","বিষয়টি","বেশ","বেশি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মতো","মতোই","মধ্যভাগে","মধ্যে","মধ্যেই","মধ্যেও","মনে","মাত্র","মাধ্যমে","মোট","মোটেই","যখন","যত","যতটা","যথেষ্ট","যদি","যদিও","যা","যাঁর","যাঁরা","যাওয়া","যাওয়ার","যাওয়া","যাকে","যাচ্ছে","যাতে","যাদের","যান","যাবে","যায়","যার","যারা","যিনি","যে","যেখানে","যেতে","যেন","যেমন","র","রকম","রয়েছে","রাখা","রেখে","লক্ষ","শুধু","শুরু","সঙ্গে","সঙ্গেও","সব","সবার","সমস্ত","সম্প্রতি","সহ","সহিত","সাধারণ","সামনে","সি","সুতরাং","সে","সেই","সেখান","সেখানে","সেটা","সেটাই","সেটাও","সেটি","স্পষ্ট","স্বয়ং","হইতে","হইবে","হইয়া","হওয়া","হওয়ায়","হওয়ার","হচ্ছে","হত","হতে","হতেই","হন","হবে","হবেন","হয়","হয়তো","হয়নি","হয়ে","হয়েই","হয়েছিল","হয়েছে","হয়েছেন","হল","হলে","হলেই","হলেও","হলো","হাজার","হিসাবে","হৈলে","হোক","হয়"]
               
def remove_stopwords(text,remove_stop_words=True):
    
    if remove_stop_words:
        
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    return text  
dataframe['article']=dataframe['article'].apply(lambda x: remove_stopwords(x))

text=dataframe['article']

    

tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(text)
sequence=tokenizer.texts_to_sequences(text)
word_index=tokenizer.word_index
num_words=len(word_index)+1
print('Found %s unique tokens.' %num_words)
max_input_len=max(len(sen) for sen in sequence)
#print(max_input_len)
X=pad_sequences(sequence,maxlen=max_input_len,padding='post')
#print('shape of data tensor:',X.shape)

dataframe['manualKeyPhrases']=dataframe['manualKeyPhrases'].apply(lambda x: x).astype(str)
dataframe['manualKeyPhrases']=dataframe['manualKeyPhrases'].apply(lambda x: remove_punctuation(x))
dataframe['manualKeyPhrases']=dataframe['manualKeyPhrases'].apply(lambda x:'sostok '+ x +'eostok ')
manualkeyphrase=dataframe['manualKeyPhrases']
t=Tokenizer(num_words=2058)
t.fit_on_texts(manualkeyphrase)
seq=t.texts_to_sequences(manualkeyphrase)
word_index1=t.word_index

num_words_output=len(word_index1)+1

print('Found %s unique tokens' %num_words_output)
max_out_len=max(len(sen) for sen in seq)
#print("Length of Longest sentence in the output: %g" %max_out_len)
Y=pad_sequences(seq,maxlen=max_out_len,padding='post')
#print('shape of data tensor:', Y.shape)

EMBEDDING_FILE='../input/bengali-word-embedding/bengali-word-embedding.txt'
#put words as dict indexes and vectors as word values
vocab_vector={}
with open(EMBEDDING_FILE,encoding='utf-8') as f:
    for line in f:
        values=line.rstrip().rsplit(' ')
        word_values=values[0]
        coefs=np.asarray(values[1:],dtype='float32')
        vocab_vector[word_values]=coefs
f.close()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
nb_words=min(max_features,len(word_index))
embedding_matrix=np.zeros((nb_words+1,embed_size))
for embed_word,v in word_index.items():
    if v>=max_features:continue
    embedding_vector=vocab_vector.get(embed_word)
    # words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector
embedding_layer=Embedding(num_words,embed_size,weights=[embedding_matrix],input_length=max_input_len)
decoder_targets_one_hot=np.zeros((
    len(text),
    max_out_len,
    num_words_output
    ),
    dtype='float32'
)

#print(decoder_targets_one_hot.shape)
for j, d in enumerate(Y):
    for t, word in enumerate(d):
        decoder_targets_one_hot[j, t, word] = 1
encoder_inputs_placeholder=Input(shape=(max_input_len,))
x=embedding_layer(encoder_inputs_placeholder)
encoder=LSTM(LSTM_NODES,return_state=True)
encoder_outputs,h,c=encoder(x)
encoder_states=[h,c]
decoder_inputs_placeholder=Input(shape=(max_out_len,))
decoder_embedding=Embedding(num_words_output,LSTM_NODES)
decoder_inputs_x=decoder_embedding(decoder_inputs_placeholder)
decoder_lstm=LSTM(LSTM_NODES,return_sequences=True,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs_x,initial_state=encoder_states)
decoder_dense=Dense(num_words_output,activation='softmax')

decoder_outputs=decoder_dense(decoder_outputs)
model=Model([encoder_inputs_placeholder,decoder_inputs_placeholder],decoder_outputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


history=model.fit([X,Y],decoder_targets_one_hot,batch_size=Batch_Size,epochs=50,validation_split=0.1,)
acc=model.evaluate([X,Y],decoder_targets_one_hot,verbose=0)

print(acc)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuarcy')
          
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


encoder_model=Model(encoder_inputs_placeholder,encoder_states)
decoder_state_input_h=Input(shape=(LSTM_NODES,))
decoder_state_input_c=Input(shape=(LSTM_NODES,))
decoder_states_inputs=[decoder_state_input_h,decoder_state_input_c]
decoder_inputs_single=Input(shape=(1,))
decoder_inputs_single_x=decoder_embedding(decoder_inputs_single)
decoder_outputs,h,c=decoder_lstm(decoder_inputs_single_x,initial_state=decoder_states_inputs)
decoder_states=[h,c]
decoder_outputs=decoder_dense(decoder_outputs)
decoder_model=Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
idx2word_input = {v:k for k, v in word_index.items()}
idx2word_target = {v:k for k, v in word_index1.items()}
def keyphrase_extraction(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_index1['sostok']
    eos = word_index1['eostok']
    output_sentence = []

    for _ in range(max_out_len):
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

    return ' '.join(output_sentence)
i = np.random.choice(len(text))
input_seq = X[i:i+1]
extraction = keyphrase_extraction(input_seq)
print('-')
print('Input:', text[i])
print('original:', manualkeyphrase[i])
print('Response:', extraction)
    
   
    