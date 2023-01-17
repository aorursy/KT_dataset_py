# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json #to import the json file
import numpy as np
import nltk
import pickle
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from  keras.optimizers import SGD
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    print("Loading exsisting data")
    
except:    
    bag=[]#to collect the bag of words
    labels=[]#to store all of the tags
    x=[]#all of the phrases stemmed into words
    y=[]#store the labels related to the phrases


    #loading the json file
    #change the location accordingly. Thanks :)
    with open('/kaggle/input/intents/intents.json') as file:
        data=json.load(file)#converting json to python dictionary

    #going through data to extract the words and  the tags
    for intent in data['intents']:
        labels.append(intent['tag'])
        for pattern in intent['patterns']:
            tokens=nltk.word_tokenize(pattern)
            bag.extend(tokens)#making a words
            x.append(tokens)#this pattern phrases split into words is used as input
            y.append(intent['tag'])#setting the labels w.r.t 'x'


    bag=[stemmer.stem(token) for token in bag if token != '?']
    bag=sorted(list(set(bag)))

    print(bag)
    #print(x)
    #print(labels)
    #print(y)

    ''' 
        From this part we start the training procedure.
            Now I will perform one hot encoding on this data to
            make it machine readable.

            I will take a numpy array(say training) of zeros same as the length of the bag. 

            Then, I will grab every token's list from 'x' and put a '1' 
            in the index corresponding to the token's originial position in the bag
    '''

    trainX=[]
    trainY=[]


    for ref,tokens in enumerate(x) :


        #print(ref,tokens)
        oheb=[]
        for word in bag:

            stem_tokens=[stemmer.stem(token) for token in tokens]

            if word in stem_tokens:
                oheb.append(1)	
            else :
                oheb.append(0)

        out=[0 for _ in range(len(labels))]
        out[labels.index(y[ref])]=1

        trainX.append(oheb)
        trainY.append(out)


    trainX=np.array(trainX)
    trainY=np.array(trainY)	
    with open("data.pickle", "wb") as f:
        pickle.dump((bag, labels, trainX, trainY), f)	

#print(trainX.shape)
#print(trainY.shape)
try:
    model=tf.keras.models.load_model('chatbot')
    print("Exsisting model loaded")
except:
    model=Sequential()
    model.add(Dense(64,input_shape=(len(trainX[0]),),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(trainY[0]),activation='softmax'))

    model.summary()

    sgd=SGD(lr=0.01,decay=1e-5,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    model.fit(trainX,trainY, epochs=1000, batch_size=8, verbose=1)
    
    model.save('chatbot')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    out=[]
    for w in words:
        if w in s_words:
            out.append(1)
        else:
            out.append(0)
    return np.array([out])


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
            
        results = model.predict([bag_of_words(inp,bag)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                print('< tag: ',tag,'>',max(results))


chat()
