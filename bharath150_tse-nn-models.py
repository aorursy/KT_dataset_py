import numpy as np 

import pandas as pd 



from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint,EarlyStopping





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.layers import BatchNormalization

import tensorflow as tf

import keras

from keras.constraints import unit_norm

from keras import regularizers

from keras import backend as K

from keras.layers import Input, Embedding,Flatten,concatenate, Conv1D, Bidirectional,Dropout

from keras.models import load_model

from numpy.testing import assert_allclose
# reading data set

data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
# removing a noisy data point

data = data[data.textID != '12f21c8f19']

data
# removing empty rows

data['text'].replace('', np.nan, inplace=True)

data.dropna(subset=['text'], inplace=True)

data.reset_index(drop=True, inplace=True)
data['text'] = data['text'].apply(lambda x: " ".join(x.split()))

data['selected_text'] = data['selected_text'].apply(lambda x: " ".join(x.split()))

data = data.astype({"text": str, "selected_text": str, 'sentiment': str})

data


x_train,x_test = train_test_split(data, test_size = 0.05, random_state=42)

x_train,x_cv = train_test_split(x_train, test_size = 0.2, random_state = 42)



print("x_train shape is", x_train.shape)

print("x_cv shape is", x_cv.shape)

print("x_test shape is", x_test.shape)

# index reset.

x_train.reset_index(inplace = True, drop = True)

x_cv.reset_index(inplace = True, drop = True)

x_test.reset_index(inplace = True, drop = True)
# https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn

fig,ax = plt.subplots(figsize = (15,5), nrows =1, ncols = 3)

ax = ax.flatten()

sns.countplot(x_train.sentiment, ax = ax[0], order = ['neutral', 'positive', 'negative'])

total = x_train.shape[0]

for p in ax[0].patches:

    height = p.get_height()

    ax[0].text(p.get_x()+p.get_width()/2.,

            height + 4,

            '{:1.2f}%'.format(height*100/total),

            ha="center") 

sns.countplot(x_cv.sentiment, ax = ax[1], order = ['neutral', 'positive', 'negative'])

total = x_cv.shape[0]

for p in ax[1].patches:

    height = p.get_height()

    ax[1].text(p.get_x()+p.get_width()/2.,

            height + 4,

            '{:1.2f}%'.format(height*100/total),

            ha="center") 

sns.countplot(x_test.sentiment, ax = ax[2], order = ['neutral', 'positive', 'negative'])

total = x_test.shape[0]

for p in ax[2].patches:

    height = p.get_height()

    ax[2].text(p.get_x()+p.get_width()/2.,

            height + 4,

            '{:1.2f}%'.format(height*100/total),

            ha="center") 
x_train
print(x_train.shape)

print(x_cv.shape)

print(x_test.shape)
#https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# tokenizing text to sequences and padding.





text_tokenizer = Tokenizer(char_level =True)

text_tokenizer.fit_on_texts(list(x_train['text']))

vocab_size_1 = len(text_tokenizer.word_index) + 1

# integer encode the documents

print("vocab size is:",vocab_size_1)



train_text = text_tokenizer.texts_to_sequences(list(x_train['text']))

cv_text = text_tokenizer.texts_to_sequences(list(x_cv['text']))

test_text = text_tokenizer.texts_to_sequences(list(x_test['text']))



train_select_text = text_tokenizer.texts_to_sequences(list(x_train['selected_text']))

cv_select_text = text_tokenizer.texts_to_sequences(list(x_cv['selected_text']))

test_select_text = text_tokenizer.texts_to_sequences(list(x_test['selected_text']))





max_length = 141 # max length of a tweet



train_text = pad_sequences(train_text, maxlen=max_length, padding='post')

cv_text =  pad_sequences(cv_text, maxlen=max_length, padding='post')

test_text = pad_sequences(test_text, maxlen = max_length, padding = 'post')









print("no. of rows sequences in train:",len(train_text))

print("no. of rows of sequences in validataion:", len(cv_text))

print("max length of sequences",max_length)
# sample datapoint

i = 1

print('text:')

print(x_train.loc[i,'text'])

print('sequence of text:')

print(text_tokenizer.texts_to_sequences([x_train.loc[i,'text']]))

print('sequence of text after padding:')

print(train_text[i])

print('select text:')

print(x_train.loc[i,'selected_text'])

print('sequence of select text:')



print(train_select_text[i])
# tokenizing sentiment.

sentiment_tokenizer = Tokenizer()

sentiment_tokenizer.fit_on_texts(x_train['sentiment'])

vocab_size_2 = len(sentiment_tokenizer.word_index) +1



train_sentiment = sentiment_tokenizer.texts_to_sequences(x_train['sentiment'])

cv_sentiment = sentiment_tokenizer.texts_to_sequences(x_cv['sentiment'])

test_sentiment = sentiment_tokenizer.texts_to_sequences(x_test['sentiment'])





print(sentiment_tokenizer.word_index)
# https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray

# creating new target variables.

def target_creation(tweets, sub_tweets):

    """

    inputs:

    tokenized tweet and tokenized selected_text.

    

    action:

    calculates start and end index of subtweet within tweet.

    

    output:

    returns start and end indices.

    

    

    """

    

    start = np.zeros(tweets.shape, dtype = 'int32')

    end = np.zeros(tweets.shape, dtype = 'int32')

    

    for i in range(tweets.shape[0]):

        

            

        a=tweets[i]

        b = sub_tweets[i]

        for j in range(len(a)):

            if (a[j:j+len(b)]==b).all():

                break



        start[i,j] = 1

        end[i,j+len(sub_tweets[i])] = 1

       

    

    return start,end
# coverting lists to array

train_select_text = np.array(train_select_text)

cv_select_text = np.array(cv_select_text)

test_select_text = np.array(test_select_text)



train_sentiment = np.array(train_sentiment)

cv_sentiment =np.array(cv_sentiment)

test_sentiment =np.array(test_sentiment)
# checking whether all created targets are correct.



train_start,train_end = target_creation(train_text,train_select_text)

count = 0

for i in range(x_train.shape[0]):



    

    if (train_text[i][np.argmax(train_start[i]):np.argmax(train_end[i])]==train_select_text[i]).all():



        count+=1

    else:

        print(len(train_text[i][np.argmax(train_start[i]):np.argmax(train_end[i])]))

        print(len(train_text[i]))

        print(len(train_select_text[i]))

        print(train_text[i][np.argmax(train_start[i]):np.argmax(train_end[i])])

        print(train_text[i])

        print(train_select_text[i])



        print(x_train.loc[i])

        print(i)



if count ==x_train.shape[0]:

    print('all targets are correct')

else:

    print(count,'targets are correct')
# checking whether all created targets are correct.



cv_start,cv_end = target_creation(cv_text,cv_select_text)

count = 0

for i in range(x_cv.shape[0]):



    

    if (cv_text[i][np.argmax(cv_start[i]):np.argmax(cv_end[i])]==cv_select_text[i]).all():



        count+=1

    else:

        print(len(cv_text[i][np.argmax(cv_start[i]):np.argmax(cv_end[i])]))

        print(len(cv_text[i]))

        print(len(cv_select_text[i]))

        print(cv_text[i][np.argmax(cv_start[i]):np.argmax(cv_end[i])])

        print(cv_text[i])

        print(cv_select_text[i])

        print(x_cv.loc[i])

        print(i)

        

    

    

if count ==x_cv.shape[0]:

    print('all targets are correct')

else:

    print(count,'targets are correct')
# checking whether all targets are correct.

test_start,test_end = target_creation(test_text,test_select_text)

count = 0

for i in range(x_test.shape[0]):



    

    if (test_text[i][np.argmax(test_start[i]):np.argmax(test_end[i])]==test_select_text[i]).all():



        count+=1

    else:

        print(len(test_text[i][np.argmax(test_start[i]):np.argmax(test_end[i])]))

        print(len(test_text[i]))

        print(len(test_select_text[i]))

        print(test_text[i][np.argmax(test_start[i]):np.argmax(test_end[i])])

        print(test_text[i])

        print(test_select_text[i])

        print(x_test.loc[i])

        print(i)



        

if count ==x_test.shape[0]:

    print('all targets are correct')

else:

    print(count,'targets are correct')
#loading char glove vectors

char2vec = {}

with open('../input/glove840b300dchar/glove.840B.300d-char.txt') as f:

    for line in f:

        values = line.split()

        char = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        char2vec[char] = coefs

        

print('no. of char vectors',len(char2vec))

print('chars covered in the model', list(char2vec.keys()))
# creating embedding matrix



vocab =text_tokenizer.word_index

embedding_matrix = np.zeros((len(vocab) + 1, 300))

for word, i in vocab.items():

    vector = char2vec.get(word)



    if vector is not None:

        embedding_matrix[i] = vector

# padding select text sequences

train_select_text = pad_sequences(train_select_text, maxlen=max_length, padding='post')

cv_select_text =  pad_sequences(cv_select_text, maxlen=max_length, padding='post')

test_select_text = pad_sequences(test_select_text, maxlen = max_length, padding = 'post')

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html





def build_model(n1,n2,n3,n4,drop,mode,bidir=False):



    """

    inputs:

    

    n1: no. of units in first layer if mode is 'lstm' else no. of filters in conv layer

    n2: no. of units in second layer if mode is 'lstm' else kernel size in conv layer

    n3: no. of neurons in first dense layer

    n4: no. of neurons in second dense layer

    mode: lstm/conv

    bidir: normal lstm or bidirectional lstm

    drop: dropout rate

    

    action:

    

    creates a neural network based on given inputs

    

    output:

    

    returns the model

    

    """

    

    keras.backend.clear_session()

    

    i1 = Input(shape=(141,), dtype='int32')

    e = Embedding(vocab_size_1, 300, weights=[embedding_matrix],  trainable=False )(i1)

    if(mode=='lstm'):

        if bidir:

            x1 = Bidirectional(keras.layers.LSTM(n1, return_sequences=True, kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001) ))(e)

            x1 = Bidirectional(keras.layers.LSTM(n2, return_sequences=True, kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001) ))(e)

        else:

            x1 = keras.layers.LSTM(n1, return_sequences=True, kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001) )(e)

            x1 = keras.layers.LSTM(n2, return_sequences=True, kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001) )(e)



    elif(mode=='conv'):

        x1=Conv1D(n1,n2,activation = 'relu',)(e)

        x1=Conv1D(n1/2,n2,activation = 'relu',)(x1)

    x1 = Flatten()(x1)

    

    i2 = Input(shape=(1,), dtype='int32')

    e = Embedding(vocab_size_2, 2, )(i2)

    x2 = Flatten()(e)

    con = concatenate([x1,x2])

    con = BatchNormalization()(con) 

    

    x1 = keras.layers.Dense(n1, activation = 'relu', kernel_initializer='he_uniform',kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001))(con)

    x1 = BatchNormalization()(x1) 

    x1 = Dropout(drop)(x1)

    x1 = keras.layers.Dense(n2, activation = 'relu', kernel_initializer='he_uniform',kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001))(x1)

    x1 = Dropout(drop)(x1)



    x2 = keras.layers.Dense(n1, activation = 'relu', kernel_initializer='he_uniform',kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001))(con)

    x2 = BatchNormalization()(x2) 

    x2 = Dropout(drop)(x2)



    x2 = keras.layers.Dense(n2, activation = 'relu', kernel_initializer='he_uniform',kernel_constraint=unit_norm(),kernel_regularizer=regularizers.l2(0.0001))(x2)

    x2 = Dropout(drop)(x2)

    

    output1 = keras.layers.Dense(141, activation = 'softmax')(x1)

    output2 = keras.layers.Dense(141, activation = 'softmax')(x2)



    model = keras.models.Model(inputs =[i1,i2], outputs = [output1,output2] )



    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipnorm=1)



    model.compile(optimizer = opt, loss = 'categorical_crossentropy' )





    return model

model = build_model(128,64,32,16,0.3,'lstm',False )

# visualizing the model

plot_model(model, show_shapes = False)

# model checkpoint to save best model.

filepath = "/kaggle/working/best_model.h5" 

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')



model.fit([train_text,train_sentiment],[train_start,train_end], validation_data = ([cv_text,cv_sentiment], [cv_start, cv_end]),epochs =20, batch_size = 32, callbacks= [checkpoint])
model = build_model(256,4,32,16,0.1,'conv' )

plot_model(model, show_shapes = False)

model.fit([train_text,train_sentiment],[train_start,train_end], validation_data = ([cv_text,cv_sentiment], [cv_start, cv_end]),epochs =20, batch_size = 32, callbacks= [checkpoint])
model = build_model(64,32,32,16,0.2,'lstm',True )

plot_model(model, show_shapes = False)

model.fit([train_text,train_sentiment],[train_start,train_end], validation_data = ([cv_text,cv_sentiment], [cv_start, cv_end]),epochs =20, batch_size = 32, callbacks= [checkpoint])
#https://stackoverflow.com/questions/51700351/valueerror-unknown-metric-function-when-using-custom-metric-in-keras

# loading the bestmodel out of three.

best_model = load_model("./best_model.h5",)
test_start,test_end= best_model.predict([test_text,test_sentiment])
# metric

def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
# calculating the jaccard score.



score = 0

for i in range(x_test.shape[0]):



    

    score = score + jaccard(x_test.text[i][np.argmax(test_start[i]):np.argmax(test_end[i])+1],x_test.selected_text[i])





    

print(score/x_test.shape[0])        
# prediction examples.



for i in range(0,x_test.shape[0],100):

    print('text:', x_test.text[i])

    print('selected text:', x_test.selected_text[i])

    print('sentiment:',x_test.sentiment[i] )

    print('predicted:',x_test.text[i][np.argmax(test_start[i]):np.argmax(test_end[i])+1])

    print('jaccard score:', jaccard(x_test.selected_text[i], x_test.text[i][np.argmax(test_start[i]):np.argmax(test_end[i])+1]))

    print('#################################\n')

    #score = score + jaccard(x_test.text[i][np.argmax(test_start[i]):np.argmax(test_end[i])+1],x_test.selected_text[i])



test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')



# tokenizing test set.

test_text = text_tokenizer.texts_to_sequences(test['text'])

test_text = pad_sequences(test_text, maxlen=max_length, padding='post')

test_sentiment = sentiment_tokenizer.texts_to_sequences(test['sentiment'])



# coverting lists to array.

test_text = np.array(test_text)

test_sentiment = np.array(test_sentiment)



# predicting using best model.



preds = []

test_start,test_end= best_model.predict([test_text,test_sentiment])



for i in range(test.shape[0]):

    preds.append(test.text[i][np.argmax(test_start[i]):np.argmax(test_end[i])])
# creating submission file.

submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

submission['selected_text'] = preds



submission.to_csv('submission.csv', index = False)