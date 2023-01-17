import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', -1)

import os

import re



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

np.random.seed(1)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test = test.sample(frac=1,random_state = 1)

train = train.sample(frac=1,random_state = 1)



train.head()
embeddings_index = {}

with open('/kaggle/input/glove100d/glove.6B.50d.txt','r',encoding = 'utf8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.array(values[1:]).astype(np.float)

        embeddings_index[word] = coefs

        

print('Found %s word vectors.' % len(embeddings_index))
print(train.info())



print(test.info())
train.drop(['keyword','location'],axis =1, inplace = True)



print(train.info())
test.drop(['keyword','location'], axis = 1, inplace = True)



print(test.info())
def clean_text(text):

    #2. remove unkonwn characrters

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

   

    #1. remove http links

    url = re.compile(r'https?://\S+|www\.\S+')

    text = url.sub(r'',text)

    

    #3,4. remove #,@ and othet symbols

    text = text.replace('#',' ')

    text = text.replace('@',' ')

    symbols = re.compile(r'[^A-Za-z0-9 ]')

    text = symbols.sub(r'',text)

    

    #5. lowercase

    text = text.lower()

    

    return text
train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))
train.head()
train_id = train['id']

train.drop(['id'],axis=1, inplace = True)



test_id = test['id']

test.drop(['id'],axis = 1, inplace = True)
word2idx = {}

new_embedding_index = {}
train_X_list = []

index = 1



embed_keys = embeddings_index.keys()

for x in train['text']:

        list1 = x.split(' ')

        new_list = []

        for i in list1:

            if((i in embed_keys)  and (i not in word2idx.keys())):

                new_embedding_index[index] = embeddings_index[i]

                word2idx[i] = index

                new_list.append(index)

                index=index+1   

                

            elif(i not in word2idx.keys()):

                new_embedding_index[index] = np.random.normal(scale=0.4, size=(50, )).astype(np.float)

                word2idx[i] = index

                new_list.append(index)

                index=index+1   



            else:

                new_list.append(word2idx[i])



        train_X_list.append(new_list)
test_X_list = []

index = len(word2idx)+1



embed_keys = embeddings_index.keys()

for x in test['text']:

        list1 = x.split(' ')

        new_list = []

        for i in list1:

            if((i in embed_keys)  and (i not in word2idx.keys())):

                new_embedding_index[index] = embeddings_index[i]

                word2idx[i] = index

                new_list.append(index)

                index=index+1   

                

            elif(i not in word2idx.keys()):

                new_embedding_index[index] = np.random.normal(scale=0.4, size=(50, )).astype(np.float)

                word2idx[i] = index

                new_list.append(index)

                index=index+1   



            else:

                new_list.append(word2idx[i])



        test_X_list.append(new_list)
print(len(new_embedding_index))
max(map(len, train_X_list)) 
max(map(len, test_X_list)) 
def pad_features(reviews_int, seq_length):

    features = np.zeros((len(reviews_int), seq_length), dtype = int)

    for i, review in enumerate(reviews_int):

        review_len = len(review)

        

        if review_len <= seq_length:

            zeroes = list(np.zeros(seq_length-review_len))

            new = zeroes+review 

        

        elif review_len > seq_length:

            new = review[0:seq_length]

        

        features[i,:] = np.array(new)

    

    return features
train_X_list = pad_features(train_X_list,55)



for i in range(3):

    extra_list =[np.array(np.zeros(55).astype(int))]

    train_X_list =  np.append(train_X_list,extra_list, axis=0)

    

print(len(train_X_list))   
train_y_list=[]

for i in train['target']:

    train_y_list.append(i)

    

for i in range(3):

    train_y_list.append(0)

print(len(train_y_list))



train_y_list=np.array(train_y_list)
test_X_list = pad_features(test_X_list,55)





extra_list =[np.array(np.zeros(55).astype(int))]





test_X_list =  np.append(test_X_list,extra_list, axis=0)
new_embedding_index[0] = np.array(np.zeros(50)).astype(np.float)
import torch

from torch.utils.data import DataLoader, TensorDataset



train_data = TensorDataset(torch.from_numpy(train_X_list),torch.from_numpy(train_y_list))





batch_size = 16

train_loader = DataLoader(train_data, batch_size = batch_size, drop_last = True)

import torch.nn as nn



class BiLSTM(nn.Module):

    #rnn for sentiment analysis

    

    def __init__(self,weights_matrix, output_size, hidden_dim,hidden_dim2, n_layers, drop_prob=0.5):

        #initialize model by setting up the layers

        super(BiLSTM, self).__init__()

        

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        #embedding and lstm layers and embedding from the glove

        num_embeddings, embedding_dim = weights_matrix.shape

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

       

        #getting values or parameters for embedding layer 

        self.embedding.weight = nn.Parameter(weights_matrix)

        

        self.lstm = nn.LSTM(embedding_dim,hidden_dim, n_layers, dropout = drop_prob, bidirectional=True, batch_first=True)

        

        

        #dropoutlayer

        self.dropout = nn.Dropout(0.3)

        

        #linear and sigmoid layers

        self.fullyconnect1 = nn.Linear(hidden_dim,hidden_dim2)

        

        self.fullyconnect2 = nn.Linear(hidden_dim2, output_size)



        #self.fullyconnect3 = nn.Linear(hidden_dim3, output_size)

        

        self.sig = nn.Sigmoid()

        

    def forward(self, x, hidden):

        #forward pass of our model 

        batch_size = x.size(0)

         

        #embedding and lstm out

        embeds = self.embedding(x)

        lstm_outs, hidden = self.lstm(embeds, hidden)

        

        # stack up lstm outputs

        lstm_outs = lstm_outs.contiguous().view(-1, self.hidden_dim)

        

        

        #dropout and fully connected layer

        out = self.dropout(lstm_outs)

        out = self.fullyconnect1(out)

        out = self.dropout(out)

        out = self.fullyconnect2(out)

        #sigmoid function

        sig_out = self.sig(out)

        

         # reshape to be batch_size first

        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        

        #return last sigmoid output and hidden state

        return sig_out, hidden

    

    def init_hidden(self, batch_size,train_on_gpu=False):

        # initialize hidden state

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,

        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

            

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda(),

                  weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())

        

        return hidden 
vals = np.array(list(new_embedding_index.values()))

vals = torch.from_numpy(vals)



output_size = 1

hidden_dim = 200

hidden_dim2 = 50

#hidden_dim3 = 50

n_layers = 2



net = BiLSTM(vals, output_size, hidden_dim,hidden_dim2, n_layers)



print(net)
train_on_gpu = True
lr=0.001



criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)





# training params



epochs =12



counter = 0

print_every = 64

clip=5 # gradient clipping



net = net.float()

# move model to GPU, if available

if(train_on_gpu):

    net.cuda()



net.train()

# train for some number of epochs

for e in range(epochs):

    # initialize hidden state

    h = net.init_hidden(batch_size)



    # batch loop

    for inputs, labels in train_loader:

        counter += 1





        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])



        # zero accumulated gradients

        net.zero_grad()



        # get the output from the model

        inputs = inputs.type(torch.LongTensor)

        inputs = inputs.cuda() 

        labels = labels.cuda()

        output, h = net(inputs, h)

        # calculate the loss and perform backprop

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()



        # loss stats

       

    print("Epoch: {}/{}...".format(e+1, epochs),

                "Loss: {:.6f}...".format(loss.item()))




test_data = torch.from_numpy(test_X_list)



test_loader = DataLoader(test_data,batch_size=batch_size)



h = net.init_hidden(batch_size)



pred = []



net.eval()

# iterate over test data

for inputs in test_loader:



    # Creating new variables for the hidden state, otherwise

    # we'd backprop through the entire training history

    h = tuple([each.data for each in h])

    

    # get predicted outputs

    inputs = inputs.type(torch.LongTensor)

    if(train_on_gpu):

        inputs = inputs.cuda()

        

    output, h = net(inputs, h)

    

    # convert output probabilities to predicted class (0 or 1)

    pred.append(torch.round(output.squeeze()))
prediction = []

for i in pred:

    prediction.append(i.tolist())



pred = []



pred = [item for sublist in prediction for item in sublist]



pred = pred[:-1] # because in test we added extra row in the last for batch size matching.



pred = [int(i) for i in pred]

print(len(pred))
output = pd.DataFrame({'id': test_id,'target': pred})



output.sort_values(["id"], axis=0, 

                 ascending=True, inplace=True)



output.to_csv('submission.csv', index=False)