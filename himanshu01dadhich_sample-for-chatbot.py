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
import nltk

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()



import numpy

import tflearn

import tensorflow

import random



import json

with open('/kaggle/input/intents.json') as file:

    data = json.load(file)
try:

    with open("data.pickle", "rb") as f:

        words, labels, training, output = pickle.load(f)

except:

    words = []

    labels = []

    docs_x = []

    docs_y = []



    for intent in data["intents"]:

        for pattern in intent["patterns"]:

            wrds = nltk.word_tokenize(pattern)

            words.extend(wrds)

            docs_x.append(wrds)

            docs_y.append(intent["tag"])



        if intent["tag"] not in labels:

            labels.append(intent["tag"])



    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    words = sorted(list(set(words)))



    labels = sorted(labels)



    training = []

    output = []



    out_empty = [0 for _ in range(len(labels))]



    for x, doc in enumerate(docs_x):

        bag = []



        wrds = [stemmer.stem(w.lower()) for w in doc]



        for w in words:

            if w in wrds:

                bag.append(1)

            else:

                bag.append(0)



        output_row = out_empty[:]

        output_row[labels.index(docs_y[x])] = 1



        training.append(bag)

        output.append(output_row)





    training = numpy.array(training)

    output = numpy.array(output)



    with open("data.pickle", "wb") as f:

        pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()



net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(output[0]), activation="softmax")

net = tflearn.regression(net)



model = tflearn.DNN(net)



try:

    model.load("model.tflearn")

except:

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

    model.save("model.tflearn")
def bag_of_words(s, words):

    bag = [0 for _ in range(len(words))]



    s_words = nltk.word_tokenize(s)

    s_words = [stemmer.stem(word.lower()) for word in s_words]



    for se in s_words:

        for i, w in enumerate(words):

            if w == se:

                bag[i] = 1

            

    return numpy.array(bag)





def chat():

    print("Start talking with the bot (type quit to stop)!")

    while True:

        inp = input("You: ")

        if inp.lower() == "quit":

            break



        results = model.predict([bag_of_words(inp, words)])

        results_index = numpy.argmax(results)

        tag = labels[results_index]



        for tg in data["intents"]:

            if tg['tag'] == tag:

                responses = tg['responses']



        print(random.choice(responses))



chat()