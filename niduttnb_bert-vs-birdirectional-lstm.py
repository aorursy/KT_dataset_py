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

import tensorflow as tf; print(tf.__version__)

from spacy.lang.en import English



import tensorflow as tf

import datetime



import pandas as pd

from spacy.lang.en import English

import spacy

import re


df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')

df.shape
df.head()
#Checking the distribution of target class

df['fraudulent'].value_counts()

#Clearly the data is imbalanced
#Check if there is any relationship between the target class and employment_type

df.pivot_table(index=['fraudulent'], columns='employment_type', aggfunc='size', fill_value=0)

#Clearly,Employment type,doesn't have any significant impact. SO we shall remove it
#Check if there is any relationship between the target class and required_experience

df.pivot_table(index=['fraudulent'], columns=['required_experience'], aggfunc='size', fill_value=0)

#Clearly,required_experience,doesn't have any significant impact. SO we shall remove it
#Check if there is any relationship between the target class and required_education

df.pivot_table(index=['fraudulent'], columns=['required_education'], aggfunc='size', fill_value=0)

#Clearly,required_educationrequired_experience,doesn't have any significant impact. SO we shall remove it
#Check if there is any relationship between the target class and required_education

df.pivot_table(index=['fraudulent'], columns=['required_education'], aggfunc='size', fill_value=0)

#Clearly,required_education,doesn't have any significant impact. SO we shall remove it
df.isna().sum()
df.fillna(' ',inplace=True)
df['text']=df['title']+" " + df['department'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits'] + " " 
df.head()
#Delete all the un-necessary Columns

delete_list=['job_id','title','location','telecommuting','has_company_logo','has_questions','department','salary_range','company_profile','description','requirements','benefits','employment_type','required_experience','required_education','industry','function']



for val in delete_list:

    del df[val]

df.head()
#Data Cleanups





df['text']=df['text'].str.replace('\n','')

df['text']=df['text'].str.replace('\r','')

df['text']=df['text'].str.replace('\t','')

  

  #This removes unwanted texts

df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',x))

df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))

  

  #Converting all upper case to lower case

df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)

  



  #Remove un necessary white space

df['text']=df['text'].str.replace('  ',' ')



  #Remove Stop words

nlp=spacy.load("en_core_web_sm")

df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))

df.head()
#Lemmenization

#Time module is just to measure the time it took as i was comparing Spacy, NLTK and Gensim. Spacy was the fastest

sp = spacy.load('en_core_web_sm')

import time

t1=time.time()

output=[]



for sentence in df['text']:

    sentence=sp(str(sentence))

    s=[token.lemma_ for token in sentence]

    output.append(' '.join(s))

df['processed']=pd.Series(output)

t=time.time()-t1

print("Time" + str(t))



        
import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

plt.figure(figsize = (20,20)) # Text that is not fraudulent(0)

wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.fraudulent == 0].processed))

plt.imshow(wc , interpolation = 'bilinear')
import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

plt.figure(figsize = (20,20)) # Text that is not fraudulent(0)

wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.fraudulent == 1].processed))

plt.imshow(wc , interpolation = 'bilinear')
import tensorflow as tf



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



vocab_size = 100000

embedding_dim = 64

max_length = 250

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size = 20000

#Tokenization



tokenizer = Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(df['processed'].values)

word_index = tokenizer.word_index

print(len(word_index))
X = tokenizer.texts_to_sequences(df['processed'].values)                         #Tokenize the dataset

X = pad_sequences(X, maxlen=max_length)     #Padding the dataset

Y=df['fraudulent']                                                                   #Assign the value of y  

print(Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20,random_state=41)

#I tried OverSampling to handle class imbalance, but the results were worse. You can try this as well

'''

from imblearn.over_sampling import (RandomOverSampler, 

                                    SMOTE, 

                                    ADASYN)

sampler =SMOTE(random_state=42)

X_train, y_train = sampler.fit_sample(X_train, y_train)

'''


model = tf.keras.Sequential([

        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

        tf.keras.layers.SpatialDropout1D(0.2),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

        tf.keras.layers.Dense(24, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)

X_train.shape
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

history = model.fit(X_train, y_train, epochs=10,batch_size=64, validation_split=0.1,callbacks=callbacks, verbose=1)

import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
y_predict=model.predict(X_test)

y_predict = np.around(y_predict, decimals = 0)

y_predict
from sklearn.metrics import confusion_matrix

cf=confusion_matrix(y_test,y_predict)

cf
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

c_report=classification_report(y_test,y_predict,target_names = ['0','1'])

print(c_report)
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import tokenization

def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
df.head()
X =df['text']        #Tokenize the dataset

Y=df['fraudulent']                                                                   #Assign the value of y  

print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20,random_state=41)

X_train.head()
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(X_train.values, tokenizer, max_len=160)

test_input = bert_encode(X_test.values, tokenizer, max_len=160)

train_labels = y_train.values
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)



model.save('model.h5')
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(train_history, 'accuracy')

plot_graphs(train_history, 'loss')
X_test.shape
y_predict=model.predict(test_input)

y_predict
y_predict = np.around(y_predict, decimals = 0)

y_predict
y_predict = np.around(y_predict, decimals = 0)

y_predict
from sklearn.metrics import confusion_matrix

cf=confusion_matrix(y_test,y_predict)

cf
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

c_report=classification_report(y_test,y_predict,target_names = ['0','1'])

print(c_report)