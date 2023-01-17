# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
train_data=pd.read_csv("/kaggle/input/fake-news/train.csv")
test_data=pd.read_csv("/kaggle/input/fake-news/test.csv")
train_data.head()
text_length=[len(str(text)) for text in train_data["text"]]
train_data['length']=text_length
train_data.head()
min(text_length),max(text_length),sum(text_length)/len(text_length)
sum([int(t<80) for t in text_length])
train_data["text"][train_data['length'] < 80].head(20)
train_data=train_data.drop(train_data['text'][train_data['length']<80].index,axis=0)
train_data.isnull().sum()
train_data["lower_text"]=[str(text).lower() for text in train_data['text']]
train_data.head()
text_train,text_val,y_train,y_val=train_test_split(train_data['lower_text'],train_data['label'],test_size=0.25)
stopwords=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
sentences=[]
for sentence in text_train:
    sentence=sentence.lower()
    text=sentence.strip().split()
    for word in text:
        while 1:
            if word in stopwords and word in text:
                text.remove(word)
            else:
                break
    sentences.append(" ".join(text))
sentences[:2]
vocab=2000
tokenizer=Tokenizer(num_words=vocab,lower=True,oov_token="<OOV>",filters='`~!@#$%^&*()_+-=][{}\|";:/.,<>?"]')
tokenizer.fit_on_texts(sentences)
train_sequences=tokenizer.texts_to_sequences(text_train)
padded=pad_sequences(train_sequences,padding='post',maxlen=4600,truncating='post')
word_index=tokenizer.word_index
print(word_index)
val_sequences=tokenizer.texts_to_sequences(text_val)
val_padded=pad_sequences(val_sequences,padding='post',maxlen=4600,truncating='post')
nan_id=[i for i in range(len(test_data['text'])) if type(test_data['text'][i])==float]
for i in nan_id:
    test_data['text'][i]="<OOV>"
test_data["lower_text"]=[str(text).lower() for text in test_data['text']]
test_data.head()
test_sequences=tokenizer.texts_to_sequences(test_data['lower_text'])
test_padded=pad_sequences(test_sequences,padding='post',maxlen=4600,truncating='post')
forward_layer=LSTM(64,activation='relu',return_sequences=True)
backward_layer=LSTM(64,activation='relu',return_sequences=True,go_backwards=True)
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab,output_dim=64,input_length=7000),
    tf.keras.layers.Bidirectional(forward_layer,backward_layer=backward_layer),
    tf.keras.layers.LSTM(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(1,activation='softmax')
])
model.summary()
model.compile(loss='binary_crossentropy',
             optimizer='RMSprop',
             metrics=['acc'])
model.fit(padded,y_train,epochs=1,validation_data=(val_padded,y_val))
model2=tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab,output_dim=64,input_length=4600),
    tf.keras.layers.Bidirectional(LSTM(32,activation='relu')),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='softmax')
])
model2.summary()
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
history2=model2.fit(padded,y_train,epochs=1)