# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import cv2

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
# atfter loading the BERT in the input-directory install BERT 

!pip install /kaggle/input/bert-for-tf2/py-params-0.8.2/py-params-0.8.2/

!pip install /kaggle/input/bert-for-tf2/params-flow-0.7.4/params-flow-0.7.4/

!pip install /kaggle/input/bert-for-tf2/bert-for-tf2-0.13.2/bert-for-tf2-0.13.2/

!pip install sentencepiece
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import tensorflow as tf 

import tensorflow_hub as hub # importing tools/keras.layers etc. from internet



import keras

from tensorflow.keras.layers import Dense, Input,LeakyReLU, Dropout, Softmax

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model



import bert



import matplotlib.pyplot as plt 



import re # regular expression operations, for cleaning text techniques
# load the competition data

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# first look at the train dataset, there are 5 cloumns, only the "text" and "target" columns contains are of interest.

# the keyword-column and the location-column may be important but contain some NaN-entries

train_df = train_df.astype({"id" : int, "target" : int, "text" : str})

train_df.head(5)
# look at the test_df

test_df = test_df.astype({"id" : int, "text" : str})

test_df.head(5)
# three example of what is not a disaster

train_df[train_df["target"] == 0]["text"].head(3)# the first three df-entries-"text"-column with target ==0
# And one that is a disaster:

train_df[train_df["target"] == 1]["text"].head(3)
# helpful function for cleaning the text with regular experessions



def remove_emoji(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)



def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # remove https? links

    text = re.sub(r'#', '', text) # remove hashtags by keeping the hashtag text

    text = re.sub(r'@\w+', '', text) # remove @usernames

    text = re.sub(r'\n',' ', text) # remove line breaks

    text = re.sub('\s+', ' ', text).strip() # remove leading, trailing, and extra spaces

    #text = re.sub(r'\-','',text)

    #text = ' '.join(re.sub("[\.\,\!\?\:\;\=\/\|\'\(\)\[\]]"," ",text).split()) # remove punctuation

    #text = remove_emoji(text)

    return text



# helpful function for extract hashtags, usernames and weblinks from tweets

def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def find_usernames(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'



# function for pereproceeding the hole text

def preprocess_text(df):

    df['clean_text'] = df['text'].apply(lambda x: clean_text(x)) # cleaning the text

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x)) # extracting the hashtags

    df['usernames'] = df['text'].apply(lambda x: find_usernames(x)) # extracting the @username(s)

    df['links'] = df['text'].apply(lambda x: find_links(x)) # extracting http(s)-links

    return df 

    

# preprocessing the 'text'-column in df and extending with additional columns 

# 'clean_text', 'hashtags', 'usernames' and 'links'

train_df = preprocess_text(train_df)

test_df = preprocess_text(test_df)

print(train_df)
#there are missing values in training set in 'keyword'- and 'location'-columns

train_df.isnull().sum()
train_df.fillna(' ')

test_df.fillna(' ')

#train_df['text_final'] = train_df['clean_text']+' '+ train_df['keyword']+' '+ train_df['location']+' '+ train_df['hashtags']

#test_df['text_final'] = test_df['clean_text']+' '+ test_df['keyword']+' '+ test_df['location']+' '+ train_df['hashtags']

train_df['text_final'] = train_df['clean_text']+' '+ train_df['keyword']#+' '+ train_df['hashtags']

test_df['text_final'] = test_df['clean_text']+' '+ test_df['keyword']#+' '+ train_df['hashtags']



train_df['lowered_text'] = train_df['text_final'].str.lower()

test_df['lowered_text'] = test_df['text_final'].str.lower()

# encoding text for bert in an bert-compatible format like: [CLS]..text..[SEP][PAD][PAD] etc.

#  cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]'=[0], mask_token='[MASK]',

# pass the text, the tokenizer from BERT an a max_len of the sequences 



def bert_encode(texts, tokenizer, max_len):  # length of encoded sequences 

    # prepare empty np-arrays for the token, mask and segments

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:  # for every text-sequence

        text = tokenizer.tokenize(text)# transform text-sequence into token-sequence

          

        text = text[:max_len-2]# cut the token-sequence at the end

        

        input_sequence = ["[CLS]"] + text + ["[SEP]"] # insert [CLS]-token at the beginning of sequence and a [SEP]-token at the end

        

        pad_len = max_len - len(input_sequence) # determine the length of the [PAD]-sequences to add on short input-sequences

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) # transforms token to token-id

        tokens += [0] * pad_len # concatenate the missing space as [0]-PAD-token

       

        pad_masks = [1] * len(input_sequence) + [0] * pad_len # pad_mask of the form 11111...00000 with 111 for input, 000 for rest

        segment_ids = [0] * max_len # segment_id of the form 00000...000

        

        all_tokens.append(tokens) # concatenate the token-sequences

        all_masks.append(pad_masks) # concatenate the padding-masks

        all_segments.append(segment_ids) # concatenate the segment-ids

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments) # return all
BertTokenizer = bert.bert_tokenization.FullTokenizer

# note that the internet must be accessible for this notebook, to download the bert-layer

# load a pretrained, trainable bert-layer as Keras.layer from the tensorflow-Hub

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1",trainable=True)



vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy() # 

to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# encoding the input- und test-features for BERT

train_input = bert_encode(train_df.lowered_text.values.astype(str), tokenizer, max_len=512) # final input-data

test_input = bert_encode(test_df.lowered_text.values.astype(str), tokenizer, max_len=512)# final test-data

train_labels = train_df.target.values # final target-data
# define a model by pass a bert-layer and a finite sequence-lenght as parameters

# to the function

def build_model(bert_layer, max_len): # etc. max_len=512, bert encoder works with sequences of finite lenght

    

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :] # for sentence classification, we’re only  interested 

    #in BERT’s output for the [CLS] token, 

    

    hidden1 = Dense(128, activation='relu')(clf_output) #128

    out = Dense(1, activation='sigmoid')(hidden1) 

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(bert_layer, max_len=512)

model.summary()
history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=5, #5

    batch_size=16

)



predictions1 = model.predict(test_input)

print(predictions1[0:30])
# try another model by using the google universal sentence encoder from https://tfhub.dev/google/universal-sentence-encoder/1 

embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")



embedded_xtrain = embedding(train_df['clean_text']).numpy()

embedded_xtest = embedding(test_df['clean_text']).numpy()

target = train_df["target"].to_numpy()



# prepare a support vector maschine with radial basis funtion kernels

from sklearn import svm

model2 = svm.SVR(kernel='rbf',gamma='auto')

model2.fit(embedded_xtrain,target)



predictions2 = model2.predict(embedded_xtest)

predictions2 = np.mat(predictions2)

predictions2 = predictions2.T

print(predictions2[0:30])

sequence_lenght = 512
# try another model by using the google universal sentence encoder https://tfhub.dev/google/universal-sentence-encoder-lite/2

USElite2_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5") #"https://tfhub.dev/google/universal-sentence-encoder-lite/2")



USElite2_embedded_xtrain = USElite2_embedding(train_df['clean_text']).numpy()

USElite2_embedded_xtest = USElite2_embedding(test_df['clean_text']).numpy()

USElite2_target = train_df["target"].to_numpy() # no embedding

USElite2_embedded_xtest.shape



USE_for_m4_xtrain = USElite2_embedded_xtrain

USE_for_m4_xtest = USElite2_embedded_xtest

USE_for_m4_target = USElite2_target
from sklearn.model_selection import train_test_split

USElite2_x_train, USElite2_x_test, USElite2_y_train, USElite2_y_test = train_test_split(

    USElite2_embedded_xtrain,

    USElite2_target,

    test_size=0.1,

    random_state=0,

    shuffle=True

)

print(USElite2_x_train.shape)

print(USElite2_y_train.shape)

print(USElite2_x_test.shape)

print(USElite2_y_test.shape)

print(USElite2_x_train)
def make_my_model():

    input = keras.layers.Input(shape=(sequence_lenght,1), dtype='float32')

    

    #Conv1D-layer expected shape (Batchsize,Width,Channels)

   

    next_layer = keras.layers.Conv1D(265,kernel_size = 10, activation = "relu",padding="valid",strides = 1)(input)

    next_layer = keras.layers.MaxPooling1D(pool_size=2)(next_layer)

    

    next_layer = keras.layers.Conv1D(64,kernel_size = 5, padding="valid", strides = 1)(next_layer)

    next_layer = keras.layers.LeakyReLU(alpha=0.1)(next_layer)

    next_layer = keras.layers.MaxPooling1D(pool_size=3, strides=1)(next_layer)

    

    next_layer = keras.layers.Flatten()(next_layer)

    

    next_layer = keras.layers.Dense(64)(next_layer)

    next_layer = keras.layers.LeakyReLU(alpha=0.1)(next_layer)

    

    #next_layer = keras.layers.Dropout(0.2)(next_layer)

    

    #next_layer = keras.layers.LeakyReLU(alpha=0.1)(next_layer)

    

    output = keras.layers.Dense(1, activation="sigmoid")(next_layer)

      

    return keras.Model(inputs=input, outputs=output)
# Reshaping the inputs. The conv1d-Layer needs (batchsize x lenght x dim=1)

# shape[0]=batchsize=6090, shape[1]=length=512, dim=1

USElite2_x_train = np.reshape(USElite2_x_train, (USElite2_x_train.shape[0], USElite2_x_train.shape[1],1))

USElite2_y_train = np.reshape(USElite2_y_train, (USElite2_y_train.shape[0],1))

# shape[0]=batchsize=1523, shape[1]=length=512, dim=1

USElite2_x_test = np.reshape(USElite2_x_test, (USElite2_x_test.shape[0], USElite2_x_test.shape[1],1))

USElite2_y_test = np.reshape(USElite2_y_test, (USElite2_y_test.shape[0],1))



model3 = make_my_model()

model3.compile("adam", loss = "binary_crossentropy", metrics = ["acc"])

model3.summary()



model3.fit(

    USElite2_x_train,

    USElite2_y_train,

    batch_size = 128,

    epochs = 15,

    validation_data = (USElite2_x_test,USElite2_y_test)

)



USElite2_embedded_xtest = np.reshape(USElite2_embedded_xtest, (USElite2_embedded_xtest.shape[0],USElite2_embedded_xtest.shape[1],1))

predictions3 = model3.predict(USElite2_embedded_xtest)

predictions3

print(predictions3[0:30])
# prepare a support vector maschine with radial basis function kernels

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer





model4 = svm.SVR(kernel='rbf', gamma='auto')

model4.fit(USE_for_m4_xtrain ,USE_for_m4_target)



predictions4 = model4.predict(USE_for_m4_xtest)

predictions4 = np.mat(predictions4)

predictions4 = predictions4.T

print(predictions4[0:30])
print(((0.5*predictions1+0.5*predictions2+0.1*predictions3+0.3*predictions4)*0.8)[0:30])
((0.5*predictions1+0.5*predictions2+0.1*predictions3+0.3*predictions4)*0.8).round().astype(int)
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print(submission)

submission['target'] =((0.5*predictions1+0.5*predictions2+0.1*predictions3+0.3*predictions4)*0.8).round().astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)