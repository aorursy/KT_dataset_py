import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

from tqdm import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def import_tweets(filename):

	#import data from csv file via pandas library

	tweet_dataset = pd.read_csv(filename, encoding = "ISO-8859-1", header = None)

	#the column names are based on sentiment140 dataset provided on kaggle

	tweet_dataset.columns = ['sentiment','id','date','flag','user','text']

	#delete 3 columns: flags,id,user, as they are not required for analysis

	for i in ['flag','id','user','date']: del tweet_dataset[i] # or tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)

	#in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1

	tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4,1)

	return tweet_dataset



tweet_dataset = import_tweets("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv")

x = list(tweet_dataset["text"])

y = list(tweet_dataset["sentiment"])

print("Text: ,",type(x),len(x))

print(type(x[0]))

print(type(y),len(y))

print(type(y[0]))
#the polarity of the tweet (0 = negative, 1positive)



x_class1 = []

x_class2 = []

y_class1 = []

y_class2 = []



for i in tqdm(range(len(y))):

    

    if y[i]==0: #negative

        y[i] = [1,0]

        y_class1.append(y[i])

        x_class1.append(x[i])

    elif y[i]==1: #positive

        y[i] = [0,1]

        y_class2.append(y[i])

        x_class2.append(x[i])

    else:

        print("Error")

        print(y[i])



data_size = 1000

x_class1 = x_class1[:data_size]

x_class2 = x_class2[:data_size]

y_class1 = y_class1[:data_size]

y_class2 = y_class2[:data_size]



x = x_class1 + x_class2

y = y_class1 + y_class2



print("Input: ,",type(x))

print(type(x[0]))

print("Output: ",type(y),len(y))

print(type(y[0]))

print(x[0])
import re

from string import punctuation, digits



def preprocess_tweet(tweet):

    #Preprocess the text in a single tweet

    #arguments: tweet = a single tweet in form of string 

    #convert the tweet to lower case

    tweet = tweet.lower()

    #convert all urls to sting "URL"

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

    #convert all @username to "AT_USER"

    tweet = re.sub('@[^\s]+','AT_USER', tweet)

    #correct all multiple white spaces to a single white space

    tweet = re.sub('[\s]+', ' ', tweet)

    #convert "#topic" to just "topic"

    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    

    converter = str.maketrans('', '', punctuation)

    tweet = tweet.translate(converter)

    

    converter = str.maketrans('', '', digits)

    tweet = tweet.translate(converter)

    

    return tweet



x = [preprocess_tweet(sentence) for sentence in tqdm(x)]

x = [sentence.split() for sentence in x] # list in list structure because of Word2vec
print("Input: ,",type(x),len(x))

print(type(x[0]))

print("Output: ",type(y),len(y))

print(type(y[0]))

print(y[0])
#Word2vec

min_count = 1

window = 2

output_size = 100

feature_epoch = 10

model_name = "1"



#train-test split

split_ratio = 0.2
from tqdm import tqdm



def Word2vecFeatureExtraction(input, y, min_count,window, output_size, epoch, model_name):

    from gensim.models import Word2Vec

    import multiprocessing

    from time import time  # To time our operations

    

    #create model

    model = Word2Vec(min_count = min_count,window=window,size=output_size,sample=6e-5,alpha=0.03,min_alpha=0.0007,negative=20,workers=4)

    t = time()

    model.build_vocab(input, progress_per=10000)

    vocab = list(model.wv.vocab)

    print("Vocab. length: ",len(vocab))

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    





    # remove out-of-vocabulary words

    t = time()

    all_data = []

    idx = 0

    for sentence in tqdm(input):

        new_sentence = []

        for word in sentence:

            if word in vocab:

                new_sentence.append(word)

        if len(new_sentence)==0:

            y.pop(idx)

            idx = idx + 1

        else:

            all_data.append(new_sentence)

            idx = idx + 1

            

    print("Removing is done. {} mins".format(round((time() - t) / 60, 2)))

    

    

    

    #train the Word2vec model

    t = time()

    model.train(all_data, total_examples = model.corpus_count, epochs = epoch, report_delay=1)

    model.save("epoch_" + str(epoch) + "_output_size_"+str(output_size)+"_"+ model_name + ".bin")

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    

    

    tokens_lengths = [len(model.wv[value]) for value in all_data]

    print("Maximum token length in each samples: ", max(tokens_lengths))

    

    print("Feature extraction is started:")

    features = [model.wv[value] for value in tqdm(all_data)]

    print("Word embeddings are ready!")

    #model = Word2Vec.load(model_name)

    

    return features,y





x,y = Word2vecFeatureExtraction(x, y, min_count, window, output_size, feature_epoch, model_name)
def paddedFeatures(features, target_dimension, position):

    import numpy as np

    padded_features = []

    

    if position == "right":

        for sample in tqdm(features):

            zeros = np.zeros((target_dimension-sample.shape[0],sample.shape[1]))

            padded_sample = np.concatenate((sample,zeros),axis=0)

            padded_features.append(padded_sample)

             

    if position == "left":

        for sample in features:

            zeros = np.zeros((target_dimension-sample.shape[0],sample.shape[1]))

            padded_sample = np.concatenate((zeros,sample),axis=0)

            padded_features.append(padded_sample)

            

    print("Padding is done.")

            

    return padded_features



x = paddedFeatures(x, target_dimension= 31, position = "right")
from sklearn.model_selection import train_test_split

#Data Split Train and Test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= split_ratio, random_state=42)
import torch.nn as nn

import torch

x_train = torch.tensor(x_train).float()

y_train = torch.tensor(y_train).float()

print(x_train.size(),y_train.size())
import torch.nn as nn

from time import time



class LSTM(nn.Module):

    def __init__(self, input_size=100, hidden_layer_size=5, output_size=2):

        super().__init__()

        

        self.batch = 1

        self.num_layers = 1

        self.hidden_layer_size = 5



        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first = True)



        self.linear = nn.Linear(hidden_layer_size, hidden_layer_size)

        

        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.linear3 = nn.Linear(hidden_layer_size, output_size)



        self.hidden_cell = (torch.zeros(self.num_layers,self.batch,self.hidden_layer_size),

                            torch.zeros(self.num_layers,self.batch,self.hidden_layer_size))



    def forward(self, input_seq):

        

        #print("Input Shape: ", input_seq.shape)

    

        input_seq = input_seq.view(1,31,100)

        #print("Expanded LSTM Input Shape: ", input_seq.shape)

        

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        #print("LSTM output shape: ",lstm_out.shape)

        

        #print("Linear Layer Input Shape: ",lstm_out.view(-1,self.hidden_layer_size).shape)

        predictions = self.linear(lstm_out.view(-1,self.hidden_layer_size))

        

        predictions = self.linear2(predictions)

        predictions = self.linear3(predictions)

        

        #print("Predictions shape: ",predictions.shape)

        #print("Returned values for loss", predictions[-1].shape)

        y = torch.nn.functional.softmax(predictions[-1], dim=0)

        

        return y

    



model = LSTM()

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print(model)



epochs = 100





average_loss = []

t = time()

for i in tqdm(range(epochs)):

    

    loss_list = []

    

    for k in range(y_train.shape[0]):

        #print("data: ",k)

        optimizer.zero_grad()

        

       

        model.hidden_cell = (torch.zeros(model.num_layers, model.batch, model.hidden_layer_size),

                        torch.zeros(model.num_layers, model.batch, model.hidden_layer_size))



        y_pred = model(x_train[k])

        

        

        

        

        single_loss = loss_function(y_pred, y_train[k])

        single_loss.backward()

        optimizer.step()

        loss_list.append(single_loss.item())



        loss_list.append(single_loss.item())

        """if k%100==0:

            #print("Data: ",k)

            #print("Toplam geçen süre:",round((time() - t) / 60, 2)," dakika")

            #print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

            print("Epoch: ",i)

            print("Data: ",k)

            print(y_pred)

            print(y_train[k])

            print("*************************************************")"""



    print("Epoch: ",i)

    average_loss.append(sum(loss_list)/len(loss_list))

    #checkpoint = {'model': LSTM(),'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}

    print("Average loss: ",sum(loss_list)/y_train.shape[0])

    #torch.save(checkpoint, 'Epoch {} Checkpoint.pth'.format(i))

    #print("\nModel kaydedildi.\n")

    print("*************************************************")