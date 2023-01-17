# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import random 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.metrics import classification_report

import os

from sklearn.model_selection import train_test_split



import tensorflow as tf

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/entity-annotated-corpus/ner_dataset.csv",encoding="latin1")

data.head()
data['Sentence #']=data['Sentence #'].ffill(axis = 0) 

data.head()
agg_func = lambda s: [(w,p, t) for w,p, t in zip(s["Word"].values.tolist(),

                                                       s['POS'].values.tolist(),

                                                        s["Tag"].values.tolist())]
agg_data=data.groupby(['Sentence #']).apply(agg_func).reset_index().rename(columns={0:'Sentence_POS_Tag_Pair'})

agg_data.head()
agg_data['Sentence']=agg_data['Sentence_POS_Tag_Pair'].apply(lambda sentence:" ".join([s[0] for s in sentence]))

agg_data['POS']=agg_data['Sentence_POS_Tag_Pair'].apply(lambda sentence:" ".join([s[1] for s in sentence]))

agg_data['Tag']=agg_data['Sentence_POS_Tag_Pair'].apply(lambda sentence:" ".join([s[2] for s in sentence]))
agg_data.shape
agg_data.head()
agg_data['tokenised_sentences']=agg_data['Sentence'].apply(lambda x:x.split())

agg_data['tag_list']=agg_data['Tag'].apply(lambda x:x.split())

agg_data.head()
agg_data['len_sentence']=agg_data['tokenised_sentences'].apply(lambda x:len(x))

agg_data['len_tag']=agg_data['tag_list'].apply(lambda x:len(x))

agg_data['is_equal']=agg_data.apply(lambda row:1 if row['len_sentence']==row['len_tag'] else 0,axis=1)

agg_data['is_equal'].value_counts()
agg_data=agg_data[agg_data['is_equal']!=0]
agg_data.shape
sentences_list=agg_data['Sentence'].tolist()

tags_list=agg_data['tag_list'].tolist()



print("Number of Sentences in the Data ",len(sentences_list))

print("Are number of Sentences and Tag list equal ",len(sentences_list)==len(tags_list))
tags_list[0]
tokeniser= tf.keras.preprocessing.text.Tokenizer(lower=False,filters='')



tokeniser.fit_on_texts(sentences_list)

print("Vocab size of Tokeniser ",len(tokeniser.word_index)+1) ## Adding one since 0 is reserved for padding
tokeniser.index_word[326]
encoded_sentence=tokeniser.texts_to_sequences(sentences_list)

print("First Original Sentence ",sentences_list[0])

print("First Encoded Sentence ",encoded_sentence[0])

print("Is Length of Original Sentence Same as Encoded Sentence ",len(sentences_list[0].split())==len(encoded_sentence[0]))

print("Length of First Sentence ",len(encoded_sentence[0]))

tags=list(set(data['Tag'].values))

print(tags)

num_tags=len(tags)

print("Number of Tags ",num_tags)



tags_map={tag:i for i,tag in enumerate(tags)}

print("Tags Map ",tags_map)
reverse_tag_map={v: k for k, v in tags_map.items()}
encoded_tags=[[tags_map[w] for w in tag] for tag in tags_list]

print("First Sentence ",sentences_list[0])

print('First Sentence Original Tags ',tags_list[0])

print("First Sentence Encoded Tags ",encoded_tags[0])

print("Is length of Original Tags and Encoded Tags same ",len(tags_list[0])==len(encoded_tags[0]))

print("Length of Tags for First Sentence ",len(encoded_tags[0]))
max_sentence_length=max([len(s.split()) for s in sentences_list])

print(max_sentence_length)
max_len=128

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical



padded_encoded_sentences=pad_sequences(maxlen=max_len,sequences=encoded_sentence,padding="post",value=0)

padded_encoded_tags=pad_sequences(maxlen=max_len,sequences=encoded_tags,padding="post",value=tags_map['O'])



print("Shape of Encoded Sentence ",padded_encoded_sentences.shape)

print("Shape of Encoded Labels ",padded_encoded_tags.shape)



print("First Encoded Sentence Without Padding ",encoded_sentence[0])

print("First Encoded Sentence with padding ",padded_encoded_sentences[0])

print("First Sentence Encoded Label without Padding ",encoded_tags[0])

print("First Sentence Encoded Label with Padding ",padded_encoded_tags[0])
target= [to_categorical(i,num_classes = num_tags) for i in  padded_encoded_tags]

print("Shape of Labels  after converting to Categorical for first sentence ",target[0].shape)
from sklearn.model_selection import train_test_split

X_train,X_val_test,y_train,y_val_test = train_test_split(padded_encoded_sentences,target,test_size = 0.3,random_state=42)

X_val,X_test,y_val,y_test = train_test_split(X_val_test,y_val_test,test_size = 0.2,random_state=42)

print("Input Train Data Shape ",X_train.shape)

print("Train Labels Length ",len(y_train))

print("Input Test Data Shape ",X_test.shape)

print("Test Labels Length ",len(y_test))



print("Input Validation Data Shape ",X_val.shape)

print("Validation Labels Length ",len(y_val))
#print("First Sentence in Training Data ",X_train[0])

#print("First sentence Label ",y_train[0])

print("Shape of First Sentence -Train",X_train[0].shape)

print("Shape of First Sentence Label  -Train",y_train[0].shape)
from tensorflow.keras import Model,Input

from tensorflow.keras.layers import LSTM,Embedding,Dense

from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
embedding_dim=128

vocab_size=len(tokeniser.word_index)+1

lstm_units=128

max_len=128



input_word = Input(shape = (max_len,))

model = Embedding(input_dim = vocab_size+1,output_dim = embedding_dim,input_length = max_len)(input_word)



model = LSTM(units=embedding_dim,return_sequences=True)(model)

out = TimeDistributed(Dense(num_tags,activation = 'softmax'))(model)

model = Model(input_word,out)

model.summary()
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,np.array(y_train),validation_data=(X_val,np.array(y_val)),batch_size = 32,epochs = 3)

preds=model.predict(X_test) ## Predict using model on Test Data


def evaluatePredictions(test_data,preds,actual_preds):

    print("Shape of Test Data Array",test_data.shape)

    y_actual=np.argmax(np.array(actual_preds),axis=2)

    y_pred=np.argmax(preds,axis=2)

    num_test_data=test_data.shape[0]

    print("Number of Test Data Points ",num_test_data)

    data=pd.DataFrame()

    df_list=[]

    for i in range(num_test_data):

        test_str=list(test_data[i])

        df=pd.DataFrame()

        df['test_tokens']=test_str

        df['tokens']=df['test_tokens'].apply(lambda x:tokeniser.index_word[x] if x!=0 else '<PAD>')

        df['actual_target_index']=list(y_actual[i])

        df['pred_target_index']=list(y_pred[i])

        df['actual_target_tag']=df['actual_target_index'].apply(lambda x:reverse_tag_map[x])

        df['pred_target_tag']=df['pred_target_index'].apply(lambda x:reverse_tag_map[x])

        df['id']=i+1

        df_list.append(df)

    data=pd.concat(df_list)

    pred_data=data[data['tokens']!='<PAD>']

    accuracy=pred_data[pred_data['actual_target_tag']==pred_data['pred_target_tag']].shape[0]/pred_data.shape[0]

    

    

    return pred_data,accuracy

        
pred_data,accuracy=evaluatePredictions(X_test,preds,y_test)
y_pred=pred_data['pred_target_tag'].tolist()

y_actual=pred_data['actual_target_tag'].tolist()
print(classification_report(y_actual,y_pred))
pred_data[pred_data['actual_target_tag']=="B-art"]
pred_data[pred_data['actual_target_tag']=="B-nat"]