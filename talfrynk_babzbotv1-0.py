!pip install tflearn

!pip install tensorflow==1.4



import nltk

from nltk.stem.lancaster import LancasterStemmer

import numpy as np

import random as rand

import tflearn

import tensorflow as tf

import json

import pickle


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))




stemmer = LancasterStemmer() #Stemmers take words and break them down into root forms



with open("/kaggle/input/babzbot/babz.json") as file:

    data = json.load(file)

    

try:

    with open("data.pickle", "rb") as f:

        words, labels, training, output = pickle.load(f)

    

except:

    

    print(data["intents"])

    

    words = []

    labels = []

    docsX = []

    docsY = []

    

    for intent in data["intents"]:

        for pattern in intent["patterns"]:

            wrds = nltk.word_tokenize(pattern) #Tokenizes the words - turns it into a list

            words.extend(wrds)

            docsX.append(wrds)

            docsY.append(intent["tag"])

            if intent["tag"] not in labels:

                labels.append(intent["tag"])

                

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #Loops through the elements in words, converts them to lower case and stems them

    words = sorted(list(set(words))) #Removes duplicate elements

    

    labels = sorted(labels)

    

    # Creating a bag of words - measures frequency of words as neural networks only understand numbers and not strings

    

    training = []

    output = []

    

    outEmpty = [0 for _ in range (len(labels))]

    

    for x, doc in enumerate(docsX):

        bag = []

        wrds  = [stemmer.stem(w.lower()) for w in doc if w != "?"] #Current pattern that is being used

        

        for w in words:

            if w in wrds:

                bag.append(1) #Word is present so append 1

            else:

                bag.append(0) #Word isnt present so append 0 

                

        outputRow = outEmpty[:]

        outputRow[labels.index(docsY[x])] = 1 #Look through labels, see where the tag is, and set that to 1 in the outputRow

        

        training.append(bag)

        output.append(outputRow)

        

    training = np.array((training))

    output = np.array(output)

    

    with open("data.pickle", "wb") as f:

        pickle.dump((words, labels, training, output), f) # Saves it

    

    



tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #Create a starting point for the neural network, with the length of the training data

net = tflearn.fully_connected(net, 8) #Initialize the network with 8 neurons

net = tflearn.fully_connected(net, 8) #Second hidden layer

net = tflearn.fully_connected(net, 8) #Second hidden layer

net = tflearn.fully_connected(net, len(output[0]), activation = "softmax") 

""""Create the output layer with the length of the output itself, softmax provides probability for each neuron 

for decision making ie if the network determines tag 1 then softmax gives tag 1 a higher chance of being 

selected as it is more likely to be the correct response"""

net = tflearn.regression(net)



model = tflearn.DNN(net)





model.fit(training, output, n_epoch = 1000, batch_size = 16, show_metric=True) #N_Epoch is the number of times it trains

    

model.save("model.tflearn")

    

# Making predictions



def bagOfWords(s, words):

    bag = [0 for _ in range (len(words))]

    sWords = nltk.word_tokenize(s)

    sWords = [stemmer.stem(w.lower()) for w in sWords if w != "?"]

    

    for i in sWords:

        for j, w in enumerate(words):

            if w == i:

                bag[j] = 1

            # No else needed as bag[] is filled with 0's

    

    return np.array(bag)



def chat():

    print("Start chatting! (Type quit to stop)")

    

    while True:

        inp = input("You: ")

        if inp.lower == ("You: quit"):

            break;

            

        else:

            results = model.predict([bagOfWords(inp, words)])[0]

            ansIndex = np.argmax(results)

            

            if results[ansIndex] < 0.5:

                print("Babz Bot: I'm sorry that does not compute. Please try again or ask another question.")

            else:    

                tag = labels[ansIndex]

                

                for tg in data["intents"]:

                    if tg["tag"] == tag:

                        responses = tg["responses"]

                        print("Babz Bot: " + rand.choice(responses))





chat()

                

            

        
