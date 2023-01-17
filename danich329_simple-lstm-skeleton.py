# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

# Not Needed since pandas has np already loaded. just use pd.np.whatever you need numpy for

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.sparse import coo_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import spacy libraries

from spacy.vocab import Vocab

from spacy.tokenizer import Tokenizer



# import gensim libraries

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec



# import pytorch

import torch

from torch.utils import data

from torch import nn

import torch.nn.functional as F

import torch.optim as optim



# sklearn metrics

from sklearn.metrics import accuracy_score, f1_score



# other libraries

import time

#from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import tqdm
# universal parameter settings



# identity columns that are featured in the testing data

# according to the data description of the competition

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



# columns that describe the comment

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']



# column with text data that will need to be converted for processing

TEXT_COLUMN = 'comment_text'



# column we eventually need to predict

TARGET_COLUMN = 'target'
# characters that we can ignore when tokenizating the TEXT_COLUMN

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
train_df = (

    pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

    .assign(bin_target=lambda x: x.target.apply(lambda x: 1 if x > 0.5 else 0))

)

print(train_df.shape)
train_df.head(2)
(

    train_df

    .bin_target

    .value_counts()

)
X_dev_pos = (

    train_df

    .query("bin_target == 1")

    .sample(30219, random_state=100) #53219

    .sort_values("id")

)



X_dev_neg = (

    train_df

    .query("bin_target == 0")

    .sample(38243, random_state=100) #488243

    .sort_values("id")

)



X_dev = X_dev_pos.append(X_dev_neg)

X_dev.shape # 541462
X = train_df[~train_df.id.isin(X_dev.id.values.tolist())]

X = (

    X

    .query("bin_target==1")

    .sample(76219,random_state=100)

    .append(

        X

        .query("bin_target==0")

        .sample(76219,random_state=100)

    )

)

X.shape
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

print(test_df.shape)
test_df.head(2)
# convert the glove twitter word vectors into 

# word2vec format

glove2word2vec(

    glove_input_file="../input/glove-global-vectors-for-word-representation/glove.twitter.27B.50d.txt",

    word2vec_output_file="word2vec.twitter.27B.50d.txt"

)
word_model = KeyedVectors.load_word2vec_format(

    "word2vec.twitter.27B.50d.txt",

    binary=False

)



# Make a dictionary with [word]->index 

global_word_dict = {key:index for index, key in enumerate(word_model.vocab.keys(), start=2)}
tokenizer = Tokenizer(vocab=Vocab(strings=list(word_model.vocab.keys())))

def index_sentences(sentence_list):

    indexed_sentences =list(

        map(

            lambda x: torch.LongTensor([

                global_word_dict[token.text.lower()] 

                if token.text in global_word_dict else 1 

                for token in tokenizer(x)

            ]),

            tqdm.tqdm(sentence_list)

        )

    )

    return indexed_sentences
torch_X = index_sentences(X[TEXT_COLUMN].values)

torch_Y = torch.FloatTensor(X.bin_target.values)
torch_X_dev = index_sentences(X_dev[TEXT_COLUMN].values)

torch_Y_dev = torch.FloatTensor(X_dev.bin_target.values)
class CustomLSTMLayer(nn.Module):

    def __init__(

        self, 

        input_size=200, hidden_size=200,

        num_layers=2, batch_size=256, 

        bidirectional=False, inner_dropout=0.25,

        outer_droput = [0.25, 0.25]

    ):

        super(CustomLSTMLayer, self).__init__()

        

        self.hidden_size = hidden_size

        self.input_size = input_size

        self.num_layers = num_layers

        self.batch_size = batch_size

        self.bidirectional = bidirectional



        self.lstm = nn.LSTM(

            self.input_size, self.hidden_size, 

            self.num_layers, batch_first=True,

            bidirectional=self.bidirectional, 

            dropout=inner_dropout

        )

        

    def forward(self, input):

        #seq_lengths = torch.zeros(input.shape(0), dtype=torch.long)

        

        #for i in range(batch_size):

        #    for j in range(max_seq - 1, -1, -1):

        #        if not torch.all(X[i, j] == 0):

        #            seq_lengths[i] = j + 1

        #           break

        _, (ht,_) = self.lstm(input)

        return ht[-1, :]

    

    def init_hidden_size(self):

        cell_state = torch.zeros(

            self.num_layers * (2 if self.bidirectional else 1),

            self.batch_size,

            self.hidden_size

        )

        

        hidden_state = torch.zeros(

            self.num_layers * (2 if self.bidirectional else 1),

            self.batch_size,

            self.hidden_size

        )

        

        return (hidden_state, cell_state)
class CustomEmbeddingLayer(nn.Module):

    def __init__(

        self, 

        vocab_size, embedding_size, 

        pretrained_embeddings=None, freeze=False

    ):

        super(CustomEmbeddingLayer, self).__init__()

        

        if pretrained_embeddings is None:

            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        else:

            rows, cols = pretrained_embeddings.shape

            self.embed = nn.Embedding(num_embeddings=rows, embedding_dim=cols, padding_idx=0)

            self.embed.weight.data.copy_(pretrained_embeddings)

    

        self.embed.weight.requries_grad = not freeze

        

    def forward(self, input):

        return self.embed(input)
class CustomFullyConnected(nn.Module):

    def __init__(self, hidden_size=200):

        super(CustomFullyConnected, self).__init__()

        

        self.fc1 = nn.Linear(hidden_size, 20)

        self.fc2 = nn.Linear(20, 10)

        self.fc3 = nn.Linear(10, 2)

        

    def forward(self, input):

        output = self.fc1(input)

        ouput = F.relu(output)

        output = self.fc2(output)

        ouput = F.relu(output)

        output = self.fc3(output)

        ouput = F.relu(output)

        return output
# Set up the dataloader

class SentenceDataLoader(data.Dataset):

    def __init__(self, train_data, train_labels):

        super(SentenceDataLoader, self).__init__()

        

        self.X = train_data

        self.Y = train_labels

        

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, index):

        return tuple([self.X[index], self.Y[index]])
def pad_sentences(batch):

    max_batch_length = max(list(map(lambda x: x[0].size(0), batch)))

    padded_sentences = torch.LongTensor(

        list(

            map(

                lambda x: pd.np.pad(x[0].numpy(), (0, max_batch_length-x[0].size(0)), 'constant', constant_values=0),

                batch

            )

        )

    )

    sentence_labels = torch.FloatTensor(list(map(lambda x: x[1], batch)))

    return (padded_sentences, sentence_labels)
# Rate at which comments are dropped for training

# too high can underfit

# too low can overfit

DROPOUT_RATE = 0.25



# NUMBER OF EPOCHS

# One Epoch is when an entire dataset is passed forward and backward

# through the neural network once.

EPOCHS = 30



# dimensions of the output vectors of each LSTM cell.

# Too high can overfit

# Too low can underfit

# The length of this vector reflects the number of

# Bidirectional CuDNNLSTM layers there will be

LSTM_HIDDEN_UNITS = 25





# dimensions of the densely-connected NN layer cells.

# The length of this vector reflects the number of

# Dense layers there will be

DENSE_HIDDEN_UNITS = 4 * LSTM_HIDDEN_UNITS



# The size of the vocab the LSTM uses

VOCAB_SIZE = len(global_word_dict)



# The side of the word vectors

EMBEDDING_SIZE = 50



#How big the batch size should be

BATCH_SIZE = 128



# The learning Rate

LEARNING_RATE = 0.01
model = nn.Sequential(

    CustomEmbeddingLayer(

        vocab_size=VOCAB_SIZE, 

        embedding_size=EMBEDDING_SIZE, 

        pretrained_embeddings=torch.FloatTensor(word_model.vectors) #find the correct code here

    ),

    CustomLSTMLayer(

        input_size=EMBEDDING_SIZE, hidden_size=LSTM_HIDDEN_UNITS,

        batch_size=BATCH_SIZE

    ),

    CustomFullyConnected(LSTM_HIDDEN_UNITS),

)

print(model)
train_dataset = SentenceDataLoader(torch_X, torch_Y)

train_data_loader = data.DataLoader(

    train_dataset, 

    batch_size=BATCH_SIZE, 

    collate_fn=pad_sentences,

    shuffle=True

)



val_dataset = SentenceDataLoader(torch_X_dev, torch_Y_dev)

val_data_loader = data.DataLoader(

    val_dataset, 

    batch_size=BATCH_SIZE, 

    collate_fn=pad_sentences,

    shuffle=True

)



# Set up the optimizer

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



# Set up the loss

ce_loss = nn.BCEWithLogitsLoss()



for epoch in range(EPOCHS):

    # Set the progress bar up

    progress_bar = tqdm.tqdm(

        enumerate(train_data_loader),

        total=len(train_data_loader),

    )

    

    #throw the model on the gpu

    model = model.cuda()

    

    avg_epoch_loss = []

    model.train()



    for index, batch in progress_bar:

        data_batch = batch[0]



        data_labels = torch.zeros(batch[0].size(0), 2)

        data_labels[range(batch[0].size(0)), batch[1].long()] = 1

        

        #Throw it on the gpu

        data_batch = data_batch.cuda()

        data_labels = data_labels.cuda()

        

        # Zero out the optimizer

        optimizer.zero_grad()

        

        #predict batch

        predicted = F.softmax(model(data_batch), dim=1)

        

        #Calculate the loss

        loss = ce_loss(predicted, data_labels)

        avg_epoch_loss.append(loss.item())

        loss.backward()

        

        # Update the weights

        optimizer.step()

        

        progress_bar.set_postfix(avg_loss=avg_epoch_loss[-1])

    

    model.eval()

    predicted_proba = []

    dev_targets = []

    

    for val_batch in val_data_loader:

        val_data_batch = val_batch[0]

        val_data_batch = val_data_batch.cuda()

        

        predicted = F.softmax(model(val_data_batch), dim=1)

        predicted_proba.append(predicted[:,1])

        dev_targets.append(val_batch[1])

    

    predicted_proba = torch.cat(predicted_proba, dim=0)

    dev_targets = torch.cat(dev_targets)

    predicted_labels = list(

        map(

            lambda x: 1 if x > 0.5 else 0,

            predicted_proba

            .cpu()

            .float()

            .detach()

            .numpy()

        )

    )

    

    msg = f"E[{epoch+1}] Train Loss: {pd.np.mean(avg_epoch_loss):.3f} "

    msg += f"Dev Accuracy: {accuracy_score(dev_targets.long().numpy(), predicted_labels):.3f} "

    msg += f"Dev F1: {f1_score(dev_targets.long().numpy(), predicted_labels):.3f}"

    print(msg)
