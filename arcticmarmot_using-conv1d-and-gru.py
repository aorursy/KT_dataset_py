# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#access data

train_frame = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_frame = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

print(train_frame.shape)

print(test_frame.shape)





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

maxlen = 220

max_words = 100000



x_train = train_frame['comment_text']

x_test = test_frame['comment_text']

y_train = train_frame['target']



#preprocessing

#from https://www.kaggle.com/tanreinama/pretext-lstm-tuning-v3-with-ensemble-tune

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



mapping = {

    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i", 

    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r", 

    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z","我":' I ',"你":' you ',

    "😂":' kidding ','😭':' sad ','😠':' angry ','😁':' excited ','👎':' discriminate ','👍':' great ','😄':' happy ',

    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 

    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 

    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  

    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": 

    "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 

    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 

    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 

    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 

    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", 

    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's":"this is","that'd": "that would", 

    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 

    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 

    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 

    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", 

    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",

    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 

    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",

    "trump's": "trump is", "obama's": "obama is", "canada's": "canada is", "today's": "today is"}

def del_punct(x):

    for p in punct:

        if p in x:

            x.replace(p,' ')

    return x

def fix_quote(x):

    x_list = x.split()

    for ele in x_list:

        if ele.startswith("'"):

            x.replace(ele,ele[1:],1)

        if ele.endswith("'"):

            x.replace(ele,ele[:-1],1)

    return x

def trans_map(x):

    for k,v in mapping.items():

        if k in x:

            x.replace(k,v)

    return x

def preprocessing(x):

    x = x.apply(trans_map)

    x = x.apply(fix_quote)

    x = x.apply(del_punct)

    return x

x_train = preprocessing(x_train)

x_test = preprocessing(x_test)

print('preprocessing completed')



tokenizer = Tokenizer(num_words = max_words)    

tokenizer.fit_on_texts(list(x_train)+list(x_test))

print('tokenizer completed')

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train,maxlen = maxlen)

x_test = pad_sequences(x_test,maxlen =maxlen)

word_index = tokenizer.word_index





glove_dir = '../input/glove6b-300d/glove.6B.300d.txt'

embeddings_index = {}

count = 0

unfound = []

with open(glove_dir) as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:],dtype = 'float32')

        embeddings_index[word] = coefs

print('embeddings_index completed')

embedding_dim = 300

embedding_matrix = np.zeros((max_words,embedding_dim))

for word,i in word_index.items():

    if i < max_words:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

        else:

            unfound.append(i)



print('matrix completed')



reverse_dict = dict([(v,k) for k,v in word_index.items()])

print(len(unfound))

    
from keras import models

from keras import layers

from keras import callbacks

from keras import Input





def get_model():

    model = models.Sequential()

    model.add(layers.Embedding(100000,300,input_length = 220,weights = [embedding_matrix],trainable = False))

    model.add(layers.Conv1D(16,8,activation = 'relu'))

    model.add(layers.MaxPool1D(3))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(16,8,activation = 'relu'))

    model.add(layers.MaxPool1D(2))

    model.add(layers.BatchNormalization())

    model.add(layers.Bidirectional(layers.GRU(128)))

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1,activation = 'sigmoid'))

    model.summary()

    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['acc'])

    return model

def get_model2():

    input_tensor = Input(shape = (220,))

    emb = layers.Embedding(100000,300,input_length = 220,weights = [embedding_matrix],trainable = False)(input_tensor)

    x = layers.Conv1D(16,8,activation = 'relu')(emb)

    x = layers.MaxPool1D(3)(x)

    x = layers.Conv1D(16,8,activation = 'relu')(x)

    x = layers.MaxPool1D(3)(x)

    x = layers.Bidirectional(layers.CuDNNLSTM(128,return_sequences = True))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(300,1,padding = 'same')(x)

    y = layers.concatenate([emb,x],axis = 1)

    y = layers.Bidirectional(layers.CuDNNLSTM(128))(y)

    y = layers.BatchNormalization()(y)

    output = layers.Dense(1,activation = 'sigmoid')(y)

    model = models.Model(input_tensor,output)

    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['acc'])

    model.summary()

    return model

def get_model3():

    input_tensor = Input(shape = (220,))

    emb = layers.Embedding(100000,300,input_length = 220,weights = [embedding_matrix],trainable = False)(input_tensor)

    conv = layers.Conv1D(128,2,activation = 'relu',padding = 'same')(emb)

    conv = layers.MaxPool1D(5)(conv)

    conv = layers.BatchNormalization()(conv)

    conv = layers.Conv1D(256,3,activation = 'relu',padding = 'same')(conv)

    conv = layers.MaxPool1D(5)(conv)

    conv = layers.BatchNormalization()(conv)

    lstm = layers.Bidirectional(layers.CuDNNLSTM(128,return_sequences = True))(emb)

    lstm = layers.Bidirectional(layers.CuDNNLSTM(128))(lstm)

    lstm = layers.BatchNormalization()(lstm)

    add = layers.add([conv,lstm])

    output = layers.Flatten()(add)

    output = layers.Dense(128,activation = 'relu')(output)

    output = layers.Dense(1,activation = 'sigmoid')(output)

    model = models.Model(input_tensor,output)

    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['acc'])

    model.summary()

    return model

def lr_func(epoch):

    return 0.01*(10**(-epoch))

model = get_model3()

model_check_point = callbacks.ModelCheckpoint('./model1_5.h5',monitor = 'val_loss',save_best_only = True)

lr_shedule = callbacks.LearningRateScheduler(lr_func)

history = model.fit(x_train[100000:],y_train[100000:]>0.5,epochs = 8,batch_size = 2048,validation_data = [x_train[:100000],y_train[:100000]>0.5],callbacks = [model_check_point])



import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()

#from sklearn.metrics import roc_auc_score

#y_test = model.predict(x_train[:100000])

#roc_auc_score(y_train[:100000]>0.5,y_test)
model.load_weights('./model1_5.h5')

result = model.predict(x_test)

result = pd.DataFrame({'id':sub.id,'prediction':result[:,0]})

result.to_csv("submission.csv", index=False, columns=['id', 'prediction'])

print('submit successful')