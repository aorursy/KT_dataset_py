import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.nn import functional as F



import sys

from torchtext import data

from torchtext import datasets

from torchtext.vocab import Vectors, GloVe

from sklearn import preprocessing

import random

from torchtext.vocab import Vectors
#Reproducing same results

SEED = 2019



#Torch

torch.manual_seed(SEED)



#Cuda algorithms

torch.backends.cudnn.deterministic = True  
vectors = Vectors(name='../input/glove6b/glove.6B.300d.txt')

vectors.dim
TEXT = data.Field(tokenize='spacy', lower=True,batch_first=True,include_lengths=True,fix_length=200,sequential=True)

LABEL = data.LabelField(dtype = torch.float,batch_first=True,) 



fields = [(None, None), ('text',TEXT),(None,None),('sentiment', LABEL)]



#loading custom dataset

training_data=data.TabularDataset(path = '../input/tweet-sentiment-extraction/train.csv',format = 'csv',fields = fields,skip_header = True)



#print preprocessed text

print(vars(training_data.examples[0]))
train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
#initialize glove embeddings

TEXT.build_vocab(train_data,min_freq=3,vectors =vectors)  

LABEL.build_vocab(train_data)



#No. of unique tokens in text

print("Size of TEXT vocabulary:",len(TEXT.vocab))



#No. of unique tokens in label

print("Size of LABEL vocabulary:",len(LABEL.vocab))



#Commonly used words

print(TEXT.vocab.freqs.most_common(10))  



#Word dictionary

print(TEXT.vocab.stoi)
#check whether cuda is available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  



#set batch size

BATCH_SIZE = 64



#Load an iterator

train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_data, valid_data), 

    batch_size = BATCH_SIZE,

    sort_key = lambda x: len(x.text),

    sort_within_batch=True,

    device = device)
class AttentionModel(torch.nn.Module):  ## General attention

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):

        super(AttentionModel, self).__init__()



        """

        Arguments

        ---------

        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator

        output_size : 3 = (pos, neg,neutral)

        hidden_sie : Size of the hidden_state of the LSTM

        vocab_size : Size of the vocabulary containing unique words

        embedding_length : Embeddding dimension of GloVe word embeddings

        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 



        --------



        """



        self.batch_size = batch_size

        self.output_size = output_size

        self.hidden_size = hidden_size

        self.vocab_size = vocab_size

        self.embedding_length = embedding_length



        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)

        self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)

        self.lstm = nn.LSTM(embedding_length, hidden_size)

        self.label = nn.Linear(hidden_size, output_size)

        #self.attn_fc_layer = nn.Linear()



    def attention_net(self, lstm_output, final_state):



        """ 

        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding

        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.



        Arguments

        ---------



        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.

        final_state : Final time-step hidden state (h_n) of the LSTM



        ---------



        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the

                  new hidden state.



        Tensor Size :

                    hidden.size() = (batch_size, hidden_size)

                    attn_weights.size() = (batch_size, num_seq)

                    soft_attn_weights.size() = (batch_size, num_seq)

                    new_hidden_state.size() = (batch_size, hidden_size)



        """



        hidden = final_state.squeeze(0)

        #print("++++",hidden.unsqueeze(2).shape)

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)



        return new_hidden_state



    def forward(self, input_sentences):



        """ 

        Parameters

        ----------

        input_sentence: input_sentence of shape = (batch_size, num_sequences)

        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)



        Returns

        -------

        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.

        final_output.shape = (batch_size, output_size)



        """



        input = self.word_embeddings(input_sentences) #m,200,300

        input = input.permute(1, 0, 2)  #200,m,300



        if batch_size is None:

            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) #1,m,128

            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) #1,m,128

        else:

            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())



        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 

        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

        #print("--",output.size(),final_hidden_state.shape)

        attn_output = self.attention_net(output, final_hidden_state)

        logits = self.label(attn_output)



        return logits
def clip_gradient(model, clip_value):

    params = list(filter(lambda p: p.grad is not None, model.parameters()))

    for p in params:

        p.grad.data.clamp_(-clip_value, clip_value)

    

def train_model(model, train_iter, epoch):

    total_epoch_loss = 0

    total_epoch_acc = 0

    model.cuda()

    

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    steps = 0

    model.train()

    for idx, batch in enumerate(train_iter):

        

        text = batch.text[0]

        target = batch.sentiment.long()

     

        if torch.cuda.is_available():

            text = text.cuda()

            target = target.cuda()

            

        if (text.size()[0] is not 64):# One of the batch returned by BucketIterator has length different than 64.

            continue

        

        optim.zero_grad()

        prediction = model(text)

        

        #print(prediction.shape,target.shape)

        loss = loss_fn(prediction, target)

        

        num_corrects = (torch.max(prediction, 1)[1].data == target.squeeze()).float().sum()

        acc = 100.0 * num_corrects/len(batch)

        loss.backward()

        clip_gradient(model, 1e-1)

        optim.step()

        steps += 1

        

        if steps % 500 == 0:

            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        

        total_epoch_loss += loss.item()

        total_epoch_acc += acc.item()

        

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)



def eval_model(model, val_iter):

    total_epoch_loss = 0

    total_epoch_acc = 0

    model.eval()

    

    with torch.no_grad():

        for idx, batch in enumerate(val_iter):

            text = batch.text[0]

            target = batch.sentiment.long()

            

            if (text.size()[0] is not 64):

                continue

            

            if torch.cuda.is_available():

                text = text.cuda()

                target = target.cuda()

                

            prediction = model(text)

            loss = loss_fn(prediction, target)

            num_corrects = (torch.max(prediction, 1)[1].data == target.squeeze()).float().sum()

            acc = 100.0 * num_corrects/len(batch)

            

            total_epoch_loss += loss.item()

            total_epoch_acc += acc.item()



    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
#define hyperparameters

learning_rate = 2e-5

batch_size = 64

output_size = 3

hidden_size = 128

embedding_length = 300



model = AttentionModel(batch_size, output_size, hidden_size, len(TEXT.vocab), embedding_length, TEXT.vocab.vectors)

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(10):

    train_loss, train_acc = train_model(model, train_iterator, epoch)

    val_loss, val_acc = eval_model(model, valid_iterator)

    

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
class SelfAttention(nn.Module):

	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):

		super(SelfAttention, self).__init__()



		"""

		Arguments

		---------

		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator

		output_size : 3 = (pos, neg,neutral)

		hidden_sie : Size of the hidden_state of the LSTM

		vocab_size : Size of the vocabulary containing unique words

		embedding_length : Embeddding dimension of GloVe word embeddings

		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

		

		--------

		

		"""



		self.batch_size = batch_size

		self.output_size = output_size

		self.hidden_size = hidden_size

		self.vocab_size = vocab_size

		self.embedding_length = embedding_length

		self.weights = weights



		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)

		self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)

		self.dropout = 0.8

		self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)

		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper

		self.W_s1 = nn.Linear(2*hidden_size, 350)

		self.W_s2 = nn.Linear(350, 30)

		self.fc_layer = nn.Linear(30*2*hidden_size, 2000)

		self.label = nn.Linear(2000, output_size)



	def attention_net(self, lstm_output):



		"""

		Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an

		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 

		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 

		connected layer of size 2000 which will be connected to the output layer of size 3 returning logits for our three classes i.e., 

		pos & neg ,neutral.

		Arguments

		---------

		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.

		---------

		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give

				  attention to different parts of the input sentence.

		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)

					  attn_weight_matrix.size() = (batch_size, 30, num_seq)

		"""

		attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))

		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)



		return attn_weight_matrix



	def forward(self, input_sentences, batch_size=None):



		""" 

		Parameters

		----------

		input_sentence: input_sentence of shape = (batch_size, num_sequences)

		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		

		Returns

		-------

		Output of the linear layer containing logits for pos & neg class.

		

		"""



		input = self.word_embeddings(input_sentences)

		input = input.permute(1, 0, 2)

        

		if batch_size is None:

			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())

			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())

		else:

			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())



		output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))

		output = output.permute(1, 0, 2)

		# output.size() = (batch_size, num_seq, 2*hidden_size)

		# h_n.size() = (1, batch_size, hidden_size)

		# c_n.size() = (1, batch_size, hidden_size)

		attn_weight_matrix = self.attention_net(output)

		# attn_weight_matrix.size() = (batch_size, r, num_seq)

		# output.size() = (batch_size, num_seq, 2*hidden_size)

		hidden_matrix = torch.bmm(attn_weight_matrix, output)

		# hidden_matrix.size() = (batch_size, r, 2*hidden_size)

		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.

		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))

		logits = self.label(fc_out)

		# logits.size() = (batch_size, output_size)



		return logits




#define hyperparameters

learning_rate = 2e-5

batch_size = 64

output_size = 3

hidden_size = 128

embedding_length = 300



model = SelfAttention(batch_size, output_size, hidden_size, len(TEXT.vocab), embedding_length, TEXT.vocab.vectors)

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(10):

    train_loss, train_acc = train_model(model, train_iterator, epoch)

    val_loss, val_acc = eval_model(model, valid_iterator)

    

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')