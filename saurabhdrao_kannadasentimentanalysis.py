# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/kannadadataset/kannadaDataset/"))

# Any results you write to the current directory are saved as output.

path = "../input/kannadadataset/kannadaDataset/"

neg_path = path + "kanNeg.txt"

pos_path = path + "kanPos.txt"



pos = []

neg=[]

labels=[]

for line in open(pos_path):

    pos.append(line)

    labels.append(1)

for line in open(neg_path):

    neg.append(line)

    labels.append(0)

print(neg[0])

print(pos[0])

data = pos+neg

df = pd.DataFrame()

df['x']=data

df['y']=labels

df = shuffle(df)

df=df.values

x=df[0:,0]

y=df[0:,1]

print(x[0],y[0])
import re



stopwordList = ['ಈ', 'ಆದರೆ', 'ಎಂದು', 'ಅವರ', 'ಮತ್ತು', 'ಎಂಬ', 'ಅವರು', 'ಒಂದು', 'ಬಗ್ಗೆ', 'ಆ', 'ಇದೆ', 'ಇದು', 'ನಾನು', 'ಮೂಲಕ', 'ನನ್ನ', 'ಅದು', 'ಮೇಲೆ', 'ಈಗ', 'ಹಾಗೂ', 'ಇಲ್ಲ', 'ಮೊದಲ', 'ನನಗೆ', 'ಹೆಚ್ಚು','ಅವರಿಗೆ', 'ತಮ್ಮ', 'ಮಾಡಿ', 'ನಮ್ಮ', 'ಮಾತ್ರ', 'ದೊಡ್ಡ', 'ಅದೇ', 'ಕೂಡ', 'ಸಿನಿಮಾ', 'ಯಾವುದೇ', 'ಯಾವ', 'ಆಗ', 'ತುಂಬಾ', 'ನಾವು', 'ದಿನ', 'ಬೇರೆ', 'ಅವರನ್ನು', 'ಎಲ್ಲಾ', 'ನೀವು', 'ಸಾಕಷ್ಟು','ಕನ್ನಡ'

, 'ಹೊಸ', 'ಮುಂದೆ', 'ಹೇಗೆ', 'ನಂತರ', 'ಇಲ್ಲಿ', 'ಕೆಲಸ', 'ಅಲ್ಲ', 'ಬಳಿಕ', 'ಒಳ್ಳೆಯ', 'ಹಾಗಾಗಿ', 'ಒಂದೇ', 'ಜನ', 'ಅದನ್ನು', 'ಬಂದ', 'ಕಾರಣ', 'ಅವಕಾಶ', 'ವರ್ಷ', 'ನಿಮ್ಮ', 'ಇತ್ತು', 'ಚಿತ್ರ', 'ಹೇಳಿ',

 'ಮಾಡಿದ', 'ಅದಕ್ಕೆ', 'ಆಗಿ', 'ಎಂಬುದು', 'ಅಂತ', '2', 'ಕೆಲವು', 'ಮೊದಲು', 'ಬಂದು', 'ಇದೇ', 'ನೋಡಿ', 'ಕೇವಲ', 'ಎರಡು', 'ಇನ್ನು', 'ಅಷ್ಟೇ', 'ಎಷ್ಟು', 'ಚಿತ್ರದ', 'ಮಾಡಬೇಕು', 'ಹೀಗೆ', 'ಕುರಿತು',

'ಉತ್ತರ', 'ಎಂದರೆ', 'ಇನ್ನೂ', 'ಮತ್ತೆ', 'ಏನು', 'ಪಾತ್ರ', 'ಮುಂದಿನ', 'ಸಂದರ್ಭದಲ್ಲಿ', 'ಮಾಡುವ', 'ವೇಳೆ', 'ನನ್ನನ್ನು', 'ಮೂರು', 'ಅಥವಾ', 'ಜೊತೆಗೆ', 'ಹೆಸರು', 'ಚಿತ್ರದಲ್ಲಿ']



def clean_text(text):

    text = text.strip()

    text = re.sub(r'[a-zA-Z]',r'',text)

    text=re.sub(r'(\d+)',r'',text)

    text=text.replace(u',','')

    text=text.replace(u'"','')

    text=text.replace(u'(','')

    text=text.replace(u'-','')

    text=text.replace(u'_','')

    text=text.replace(u'&','')

    text=text.replace(u'/','')

    text=text.replace(u')','')

    text=text.replace(u'"','')

    text=text.replace(u':','')

    text=text.replace(u"'",'')

    text=text.replace(u"‘‘",'')

    text=text.replace(u"’’",'')

    text=text.replace(u"''",'')

    text=text.replace(u".",'')

    text=text.replace(u'#','')

    text = text.replace(u'!','')

    text = text.replace(u'@','')

    text = text.replace(u'?','')

    text = re.sub('https?://[a-zA-Z0-9./]+',' ',text) #links

    text = re.sub(r'@[a-zA-Z0-9]+',' ',text) #mentions

    texts = text.split()

    newText=[]

    for text in texts:

        if text not in stopwordList:

            newText.append(text)

    return " ".join(newText)



x = [clean_text(sample) for sample in x]

print(x[0:5])
import pickle

num_words = 70

BOW = set()



for line in x:

    words = line.split()

    for word in words:

        BOW.add(word)

BOW = list(BOW)

BOW_len = len(BOW) + 1



print(BOW,BOW_len)



word_index = { BOW[ptr-1]:ptr for ptr in range(1,len(BOW)+1) }  

word_index["<PAD>"] = 0

reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])

del BOW

newX = []



for line in x:

    t=[]

    words = line.split()

    for word in words:

        t.append(word_index[word])

    if len(t) < num_words:

        t+= [0]*(num_words-len(t))

    newX.append(t)

newX = np.array(newX)

print(newX)

print(newX.shape)



filePath = "word_index.pkl"

fileout = open(filePath,'wb')

pickle.dump(word_index,fileout)

fileout.close()
from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer

x=newX



SPLIT = 80

limit = len(x)*SPLIT//100



xtrain = x[:limit]

ytrain = y[:limit]

xtest = x[limit:]

ytest = y[limit:]



train_len = len(xtrain)

test_len = len(xtest)

print(np.array(x))

print(y)



print(xtrain.shape)
from keras.layers import Dropout

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D,MaxPooling1D

from keras.layers.embeddings import Embedding

model = Sequential()

model.add(Embedding(BOW_len,128,input_length=num_words))

model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling1D(2))

model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling1D(2))

model.add(LSTM(128))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

h = model.fit(xtrain,ytrain,epochs=20,batch_size=64,validation_data=(xtest,ytest),verbose=1)

model.save('model.h5')
plt.plot(h.history['acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.title('Model Loss')

plt.show()
word_index = pickle.load(open('word_index.pkl','rb'))



samples  = df[0:8,0]

trueOutput= df[0:8,1]



samples  = ["ಅವನು ಒಳ್ಳೆಯ ಮನುಷ್ಯ","ವ್ಯಾಯಾಮ ಆರೋಗ್ಯಕ್ಕೆ ಒಳ್ಳೆಯದು","ಅವನು ಕೆಟ್ಟವನು","ವೇಗವಾಗಿ ಚಾಲನೆ ಮಾಡುವುದು ಅಪಾಯಕಾರಿ"] + list(samples)

trueOutput= [1,1,0,0]+list(trueOutput)



def conv2Test(text):

    text=clean_text(text)

    text=text.split()

   # text=[word_index[x] for x in text]

    newText=[]

    for x in text:

        if x in word_index:

            newText.append(word_index[x])

        else:newText.append(0)

    text=newText

    text+=[0]*(num_words-len(text))

    return np.array(text)



def result(prob):

    if prob > 0.5:print("Actual Result: Positive",prob)

    else: print("Actual Result: Negative",prob)



i=0

for text in samples:

    print("-"*100)

    print(text)

    sample = conv2Test(text).reshape((1,num_words))

    a=model.predict(sample)[0]

    result(a)

    print("Expected Result:",trueOutput[i])

    i+=1