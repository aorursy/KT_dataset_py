# Inspired from toxication-with-embeddings-and-keras-lstm. https://www.kaggle.com/samarthsarin/toxication-with-embeddings-and-keras-lstm



import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))





from keras.models import Sequential,Model

from keras.layers import Embedding,Input,Activation,Flatten,CuDNNLSTM,Dense,Dropout,Bidirectional

from keras.layers import Convolution1D,GlobalAveragePooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LeakyReLU

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt

import re

import gc

import pickle

import seaborn as sns

%matplotlib inline

tqdm.pandas()

# Reads CSV to df.



train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')



test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
# Cleans up the Data set by decontracting contractions and removing special character/punctuations.



#Decontraction inspired from https://gist.github.com/nealrs/96342d8231b75cf4bb82

cList = {

  "ain't": "am not",

  "aren't": "are not",

  "can't": "cannot",

  "can't've": "cannot have",

  "'cause": "because",

  "could've": "could have",

  "couldn't": "could not",

  "couldn't've": "could not have",

  "didn't": "did not",

  "doesn't": "does not",

  "don't": "do not",

  "hadn't": "had not",

  "hadn't've": "had not have",

  "hasn't": "has not",

  "haven't": "have not",

  "he'd": "he would",

  "he'd've": "he would have",

  "he'll": "he will",

  "he'll've": "he will have",

  "he's": "he is",

  "how'd": "how did",

  "how'd'y": "how do you",

  "how'll": "how will",

  "how's": "how is",

  "I'd": "I would",

  "I'd've": "I would have",

  "I'll": "I will",

  "I'll've": "I will have",

  "I'm": "I am",

  "I've": "I have",

  "isn't": "is not",

  "it'd": "it had",

  "it'd've": "it would have",

  "it'll": "it will",

  "it'll've": "it will have",

  "it's": "it is",

  "let's": "let us",

  "ma'am": "madam",

  "mayn't": "may not",

  "might've": "might have",

  "mightn't": "might not",

  "mightn't've": "might not have",

  "must've": "must have",

  "mustn't": "must not",

  "mustn't've": "must not have",

  "needn't": "need not",

  "needn't've": "need not have",

  "o'clock": "of the clock",

  "oughtn't": "ought not",

  "oughtn't've": "ought not have",

  "shan't": "shall not",

  "sha'n't": "shall not",

  "shan't've": "shall not have",

  "she'd": "she would",

  "she'd've": "she would have",

  "she'll": "she will",

  "she'll've": "she will have",

  "she's": "she is",

  "should've": "should have",

  "shouldn't": "should not",

  "shouldn't've": "should not have",

  "so've": "so have",

  "so's": "so is",

  "that'd": "that would",

  "that'd've": "that would have",

  "that's": "that is",

  "there'd": "there had",

  "there'd've": "there would have",

  "there's": "there is",

  "they'd": "they would",

  "they'd've": "they would have",

  "they'll": "they will",

  "they'll've": "they will have",

  "they're": "they are",

  "they've": "they have",

  "to've": "to have",

  "wasn't": "was not",

  "we'd": "we had",

  "we'd've": "we would have",

  "we'll": "we will",

  "we'll've": "we will have",

  "we're": "we are",

  "we've": "we have",

  "weren't": "were not",

  "what'll": "what will",

  "what'll've": "what will have",

  "what're": "what are",

  "what's": "what is",

  "what've": "what have",

  "when's": "when is",

  "when've": "when have",

  "where'd": "where did",

  "where's": "where is",

  "where've": "where have",

  "who'll": "who will",

  "who'll've": "who will have",

  "who's": "who is",

  "who've": "who have",

  "why's": "why is",

  "why've": "why have",

  "will've": "will have",

  "won't": "will not",

  "won't've": "will not have",

  "would've": "would have",

  "wouldn't": "would not",

  "wouldn't've": "would not have",

  "y'all": "you all",

  "y'alls": "you alls",

  "y'all'd": "you all would",

  "y'all'd've": "you all would have",

  "y'all're": "you all are",

  "y'all've": "you all have",

  "you'd": "you had",

  "you'd've": "you would have",

  "you'll": "you you will",

  "you'll've": "you you will have",

  "you're": "you are",

  "you've": "you have"

}



c_re = re.compile('(%s)' % '|'.join(cList.keys()))



def deContract(text, c_re=c_re):

    def replace(match):

        return cList[match.group(0)]

    return c_re.sub(replace, text.lower())





def clean_special_chars(text):

    

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    for p in mapping:

        text = text.replace(p, mapping[p])

    for p in punct:

        text = text.replace(p, ' ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    return text







def preProcessing(x):

    x=deContract(x)

    x=clean_special_chars(x)

    return x

    





train["comment_text"] = train["comment_text"].progress_apply(lambda x: preProcessing(x))



test["comment_text"] = test["comment_text"].progress_apply(lambda x: preProcessing(x))
# Function for transforming target to categorical set. Values > 0.5 are considered Toxic and vice versa.

def target(value):

    if value>=0.5:

        return 1

    else:

        return 0
#Use apply function to apply target function.

train['target'] = train['target'].progress_apply(target)
#Defines x features and y target.

x = train['comment_text']

y = train['target']

# Link to the Glove Pretrained Embedding

GlovePath = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

 
#Tokenizes Text.

token = Tokenizer()

token.fit_on_texts(x)

pad_seq = pad_sequences(token.texts_to_sequences(x),maxlen = 300)

vocab_size=len(token.word_index)+1
# Function used to building the embedding matrix to load into the model.

def build_matrix(word_index, embedPath):

    with open(embedPath, 'rb') as fp:

        embedding_index = pickle.load(fp)

    

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix

# Runs the function to build the matrix.

embedding_matrix= build_matrix(token.word_index,GlovePath)

del (train)

gc.collect()
# Creates function load model. Created using bidirectional LSTM 

def BuildModel(vocab_size,embedding_Matrix):

    model = Sequential()

    model.add(Embedding(vocab_size,300,input_length = 300,weights = [embedding_matrix],trainable = False))

    model.add(Bidirectional(CuDNNLSTM(300,return_sequences=True)))

    model.add(Convolution1D(64,7,padding='same'))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])

    return model
# Splits up the training/validation sets.

x_train,x_test,y_train,y_test = train_test_split(pad_seq,y,test_size = 0.15,random_state = 42)
del (x,y)

gc.collect()

# Builds the model.

model = BuildModel(vocab_size,embedding_matrix)

model.summary()
# Collects the history of the model.

history = model.fit(x_train,y_train,epochs = 5,batch_size=1000,validation_data=(x_test,y_test))
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['loss']

training_loss = values['val_loss']

epochs = range(1,6)



#Dumps history into json file for further analysis.

import json

json = json.dumps(values)



f = open("history.json","w")

f.write(json)

f.close()
#Plots the training accuracy vs the validation accuracy.

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

#Plots the train/val loss over the epochs.

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
#Save model to a file for future use.

filename = 'finalized_model.sav'

pickle.dump(model, open(filename, 'wb'))
# Tokenizes the test comment text.

X = test['comment_text']

test_pad_seq = pad_sequences(token.texts_to_sequences(X),maxlen = 300)
# Uses the model to predict test df.

prediction = model.predict(test_pad_seq)
submission = pd.DataFrame([test['id']]).T

submission['prediction'] = prediction
submission.to_csv('submission.csv', index=False)
submission.head()