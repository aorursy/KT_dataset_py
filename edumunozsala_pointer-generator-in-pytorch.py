#Import utils and libraries

import pandas as pd

import numpy as np

import os

import pickle

from collections import Counter



from typing import Callable, Optional

from copy import deepcopy



# Import pytoch libraries and modules 

import torch

import torch.nn as nn

from torch.nn import functional as F

from torch import optim

from torch.nn.utils import clip_grad_norm_

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



#Import libraries for text procesing

import nltk

from nltk.tokenize import word_tokenize

nltk.download('punkt')

!pip install pkbar

import pkbar
!pip install rouge

from rouge import Rouge
class Parameters:

  # Model Parameters

  hidden_size: int = 150  # of the encoder; default decoder size is doubled if encoder is bidi

  dec_hidden_size: Optional[int] = 200  # if set, a matrix will transform enc state into dec state

  embed_size: int = 100 # Size of the embedding vectors

  eps=1e-31

  batch_size=16 

  enc_bidi = True #Set the encoder as bidirectional

  enc_rnn_dropout = 0.1 # Set the dropout parameter in the encoder

  enc_attn = True #Activate the encoder attention

  dec_attn = True #Activate the decoder attention

  pointer = True #Activate the pointer generator mechanism

  # Set different dropout probabilities in the decoder

  dec_in_dropout=0.1

  dec_rnn_dropout=0.1

  dec_out_dropout=0.1

  # Vocabulary and data parameters

  max_src_len: int = 65  # exclusive of special tokens such as EOS

  max_tgt_len: int = 15  # exclusive of special tokens such as EOS

  vocab_min_frequency: int = 3

  # Data paths

  embed_file: Optional[str] = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'  # use pre-trained embeddings

  data_path: str = '/kaggle/input/cleaned-news-summary/cl_train_news_summary_more.csv'

  val_data_path: Optional[str] = '/kaggle/input/cleaned-news-summary/cl_train_news_summary_more.csv'

  test_data_path: str = '/kaggle/input/cleaned-news-summary/cl_valid_news_summary_more.csv'

  # Parameters to save the model

  resume_train = False

  encoder_weights_path='encoder_sum.pt'

  decoder_weights_path='decoder_sum.pt'

  encoder_decoder_adapter_weights_path='adapter_sum.pt'

  losses_path='val_losses.pkl'

  print_every = 100
def simple_tokenizer(text, lower=False, newline=None):

  if lower:

    text = text.lower()

  if newline is not None:  # replace newline by a token

    text = text.replace('\n', ' ' + newline + ' ')

  return text.split()

class Vocab(object):

  PAD = 0

  SOS = 1

  EOS = 2

  UNK = 3



  def __init__(self):

    ''' Initialize the structuresto store the information about the vocabulary'''

    self.word2index = {}

    self.word2count = Counter()

    self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

    self.index2word = self.reserved[:]

    self.embeddings = None



  def add_words(self, words):

    ''' Add words to the vocabulary'''

    for word in words:

      #if it is an unseen word  

      if word not in self.word2index:

        #Include the word in the mapping from word to index

        self.word2index[word] = len(self.index2word)

        # Include the word in the indexes

        self.index2word.append(word)

    # Increment the count of ocurrencies of the word to 1

    self.word2count.update(words)

  

  def load_embeddings(self, file_path: str, dtype=np.float32) -> int:

    ''' Load the embedding vectors from a file into the vocabulary'''

    num_embeddings = 0

    vocab_size = len(self)

    with open(file_path, 'rb') as f:

      # For every word in the embedding vectors

      for line in f:

        line = line.split()

        word = line[0].decode('utf-8')

        # self.add_words([word])

        # Get the index of the embedded word

        idx = self.word2index.get(word)

        if idx is not None:

          # Extract the embedding vector of the word

          vec = np.array(line[1:], dtype=dtype)

          #If the embedding vector is not initialized

          if self.embeddings is None:

            # Set the embeddings dimension, initialize the embedding vector to zeros

            n_dims = len(vec)

            self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)

            self.embeddings[self.PAD] = np.zeros(n_dims)

          # Store the embedding in the array of embeddings 

          self.embeddings[idx] = vec          

          num_embeddings += 1

    return num_embeddings



  def save_to_file(self, filename):

    ''' Save the Vocab object to a file'''

    with open(filename,'wb') as f:

        pickle.dump(self,f) 



  def __getitem__(self, item):

    ''' Get the next item when iterating over the instance'''

    if type(item) is int:

      return self.index2word[item]

    return self.word2index.get(item, self.UNK)



  def __len__(self):

    ''' Return the length of the instance or vocabulary'''

    return len(self.index2word)





def load_vocab(filename):

    ''' Load a Vocab instance from a file'''

    with open(filename,'rb') as f:

        v = pickle.load(f)

    return v
class Dataset(object):

  ''' Create a Class to store the data input and its features'''

  def __init__(self, filename: str, tokenize: Callable=simple_tokenizer, max_src_len: int=None,

               max_tgt_len: int=None, max_rows: int=None, truncate_src: bool=False, truncate_tgt: bool=False):

    print("Reading dataset %s..." % filename, end=' ', flush=True)

    # Save the filename and initialize the variables

    self.filename = filename

    self.pairs = []

    self.src_len = 0

    self.tgt_len = 0

    self.max_rows = max_rows



    #Read the csv file, using max rows if it is defined

    if max_rows is None:

        df = pd.read_csv(filename, encoding='utf-8')

    else:

        df = pd.read_csv(filename, encoding='utf-8', nrows=max_rows

                        )

    # Tokenize the source texts

    sources = df['text'].apply(lambda x : tokenize(x))

    # Truncate the sources texts

    if truncate_src:

        sources = [src[:max_src_len] if len(src)>max_src_len else src for src in sources]

    # Tokenize the targets

    targets = df['summary'].apply(lambda x : tokenize(x))

    # Trucate the targets

    if truncate_tgt:

        targets = [tgt[:max_tgt_len] if len(tgt)>max_tgt_len else tgt for tgt in targets]

        

    # Calculate the length of every source and targets        

    src_length = [len(src)+1 for src in sources]

    tgt_length = [len(tgt)+1 for tgt in targets]

    #Calculate the max length of the sources and the targets

    max_src = max(src_length)

    max_tgt = max(tgt_length)

    #Create a tuple contaiing source,target,source length, target length

    self.src_len = max_src

    self.tgt_len = max_tgt

    # Insert the source text and target in the pairs class atribute

    self.pairs.append([(src, tgt, src_len, tgt_len) for src,tgt,src_len,tgt_len in zip(sources,targets,src_length,tgt_length)])

    self.pairs = self.pairs[0]

    print("%d pairs." % len(self.pairs))



  def build_vocab(self, min_freq, embed_file: str=None) -> Vocab:

    ''' Build the vocabulary extracted from the texts in the object class

        Input:

        - min_freq: integer, minimum ocurrencies needed to include the word in the vocab

        - embed_file: string, path + filename of the embeddings file

    '''

    # Extract the words in the whole corpus

    total_words=[src+tgr for src,tgr,len_src,len_tgr in self.pairs]

    total_words = [item for sublist in total_words for item in sublist]

    # Create a counter to count the ocurrencies of every word in the corpus

    word_counts = Counter(total_words)

    # Create a vocabulary object

    vocab=Vocab()

    for word,count in word_counts.items():

        # If occurencies of the word are bigger then min_freq

        if(count>min_freq):

            # Include the cord in the vocabulary

            vocab.add_words([word])  

    # Load the embeddings in the vocab object

    count = vocab.load_embeddings(embed_file)

    print("%d pre-trained embeddings loaded." % count)



    return vocab  

class MyDataset(nn.Module):

    ''' A Dataset Class where we store all the data needed during the training phase'''

    

    def __init__(self, src_sents, trg_sents, vocab):

      '''Initialize the instance and store the source texts, targets or summaries '''

      self.src_sents = src_sents

      self.trg_sents = trg_sents

      self.vocab=vocab

      # Keep track of how many data points.

      self._len = len(src_sents)



    def __getitem__(self, index):

        ''' Return the ith items from the object

            Input:

            - Index: integer, index of the items to return

            Output:

            - a dictionary with keys x the source texts, y the targets, 

              x_len length of source texts, y_len the length of targets

        '''

        return {'x':self.src_sents[index], 

                'y':self.trg_sents[index], 

                'x_len':len(self.src_sents[index]), 

                'y_len':len(self.trg_sents[index])}

    

    def __len__(self):

        ''' Return the length of the object'''

        return self._len

def tensorize(vocab, tokens):

    ''' Convert the tokens received to a tensor '''

    return torch.tensor([vocab[token] for token in tokens])



def pad_sequence(vectorized_sent, max_len):

    ''' Padding the sentence (tensor) to max_len '''

    pad_dim = (0, max_len - len(vectorized_sent))

    return F.pad(vectorized_sent, pad_dim, 'constant').tolist()



def preprocess(x,y,p,vocab):

    ''' Prepare a source text x and a target summary y: convert them to tensors,

        pads the sentences to its max length.

    '''

    # Convert x and y to tensors using the vocabulary

    tensors_src = tensorize(vocab, x)

    tensors_trg = tensorize(vocab, y) 

    # Return the padded sequence of x and y and its length

    return {'x':pad_sequence(tensors_src, p.max_src_len), #¿max_source_len?

          'y':pad_sequence(tensors_trg, p.max_tgt_len), #¿,ax_target_len?

          'x_len':len(tensors_src), 

          'y_len':len(tensors_trg)}



def sort_batch_by_len(data_dict,p,vocab):

    ''' Return a batch of sentences processed and ordered by its length

    '''

    data=[]

    res={'x':[],'y':[],'x_len':[],'y_len':[]}

    # For every x and y in the data input

    for i in range(data_dict['x_len']):

        # Preprocess and tokenize the x and y

        data.append(preprocess(data_dict['x'][i],data_dict['y'][i],p,vocab))

    # For every preprocessed text, recreate the x and y lists

    for i in range(len(data)):

        res['x'].append(data[i]['x'])

        res['y'].append(data[i]['y'])

        res['x_len'].append(len(data[i]['x']))

        res['y_len'].append(len(data[i]['y']))  

    

    # Sort indices of data in batch by lengths.

    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    # Create a batch of data ordered by its length

    data_batch = {name:[_tensor[i] for i in sorted_indices]

                  for name, _tensor in res.items()}

    return data_batch

class EncoderRNN(nn.Module):

  ''' Define an encoder in a seq2seq architecture'''

  def __init__(self, embed_size, hidden_size, bidi=True, rnn_drop: float=0):

    super(EncoderRNN, self).__init__()

    # Set the hidden size

    self.hidden_size = hidden_size

    # Activate bidirectional mode

    self.num_directions = 2 if bidi else 1

    # Define the GRU layer of the encoder

    self.gru = nn.GRU(embed_size, hidden_size, bidirectional=bidi, dropout=rnn_drop)



  def forward(self, embedded,hidden,input_lengths=None):

    ''' Run a Forward pass of the encoder to return outputs

        Input:

        - embedded: tensor, the embedding of the input data (word of the soure text)

        - hidden: a tensor, the previous hidden state of the encoder

        - input:lengths: a list of integers, length of the inputs 

    '''

    # Pack the padded sequence of the embedded input

    if input_lengths is not None:

      embedded = pack_padded_sequence(embedded, input_lengths,batch_first=True)

    

    # Apply the GRU layer of the encoder

    output, hidden = self.gru(embedded,hidden)

    

    # Pad the sequence output

    if input_lengths is not None:

      output, _ = pad_packed_sequence(output)

    # If bidirectional

    if self.num_directions > 1:

      # Transform the hidden state tensor  

      # hidden: (num directions, batch, hidden) => (1, batch, hidden * 2)

      batch_size = hidden.size(1)

      hidden = hidden.transpose(0, 1).contiguous().view(1, batch_size,

                                                        self.hidden_size * self.num_directions)

    return output, hidden



  def init_hidden(self, batch_size, device):

    ''' Initialize the hidden state of the encoder to zeros: num_directions, batch size, hidden size '''

    return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=device) 

class DecoderRNN(nn.Module):

  ''' Define a decoder with atention in a seq2seq architecture'''

  def __init__(self, vocab_size, embed_size, hidden_size, enc_attn=True, dec_attn=True,

               enc_attn_cover=True, pointer=True,

               in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None,

               epsilon: float=0.0, device: str="cpu"):

    ''' Initialize the decoder instance defining its parameters:

            Input:

                - vocab_size: integer, number of words in the vocabulary 

                - embed_size: integer, size of the embedding layer

                - hidden_size: integer, size of the hidden layer (Hyperparameter)

                - enc_attn: activate the attention in the encoder

                - dec_attn: activate the attention in the decoder

                - enc_attn_cover: activate the coverage mechanism in the attention

                - pointer: activate the pointer generation

                - in_drop: dropout probability to apply to the input of the decoder

                - rnn_drop: dropout probability to apply to the GRU layer of the decoder

                - out_drop: dropout probability to apply to the output of the decoder

                - enc_hidden_size: dimension if the hidden state of the encoder

                - epsilon: float

                - device: cpu or gpu, device to store the tensors

    '''



    super(DecoderRNN, self).__init__()

    # Set the attibutes

    self.vocab_size = vocab_size

    self.hidden_size = hidden_size

    self.combined_size = hidden_size

    self.device = device

    self.eps = epsilon

    # Define the input dropout layer 

    self.in_drop = nn.Dropout(in_drop) if in_drop > 0 else None

    # Define the GRU layer

    self.gru = nn.GRU(embed_size, hidden_size, dropout=rnn_drop)

    

    # Set the hidden size of the encoder to the hidden size of the decoder if it is not defined

    if not enc_hidden_size: enc_hidden_size = self.hidden_size

    # Bilinear layer of the encoder

    self.enc_bilinear = nn.Bilinear(hidden_size, enc_hidden_size, 1)

    

    self.combined_size += enc_hidden_size

    if enc_attn_cover:

      # Initialize the weights of the coverage mechanism

      self.cover_weight = nn.Parameter(torch.rand(1))



    # Bilinear layer of the encoder

    self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

    self.combined_size += self.hidden_size

    

    # Define the output dropout layer

    self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None

    # Define the pointer generator layer

    self.ptr = nn.Linear(self.combined_size, 1)



    # Define the linear layer at the output

    self.out = nn.Linear(self.combined_size, vocab_size)



  def forward(self, embedded, hidden, encoder_hidden=None, decoder_states=None, coverage_vector=None, *,

              encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True):

    ''' Run a Forward pass of the decoder to return outputs

        Input:

        - embedded: tensor, the embedding of the input data (decoder output in the last step

        - hidden: a tensor, the previous hidden state of the decoder

        - decoder_states: tensor, hidden state of the decoder in the last step

        - coverage_vector: tensor, coverage vector at this step

        - encoder_word_idx: tensor, indexes of the words in the source text

        - ext_vocab_size: integer, vocabulary size of the extended vocabulary

        - log_prob: bool, use of Log Softmax or Softmax in the output

    '''

    #print('Self Dev:',self.device)

    # Set the batch sze and initialize the combined context vectors

    batch_size = embedded.size(0)

    combined = torch.zeros(batch_size, self.combined_size, device=self.device)

    # Apply the dropout layer to the input data

    if self.in_drop: embedded = self.in_drop(embedded)

    # Apply the GRU layer

    output, hidden = self.gru(embedded.unsqueeze(0), hidden)  # unsqueeze and squeeze are necessary

    combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)

    offset = self.hidden_size

    enc_attn, prob_ptr = None, None  # for visualization



    # Set the encoder steps and total size

    num_enc_steps = encoder_hidden.size(0)

    enc_total_size = encoder_hidden.size(2)

    #print('Encoder hidden: ', encoder_hidden.shape,' Hidden: ', hidden.shape)

    # Apply the Bilinear layer

    enc_attn = self.enc_bilinear(hidden.expand(num_enc_steps, batch_size, -1).contiguous(),encoder_hidden)

    # print(hidden.shape,hidden.expand(num_enc_steps, batch_size, -1).shape,encoder_hidden.shape,enc_attn.shape)

    

    # Update the attention weigths

    if coverage_vector is not None:

        enc_attn += self.cover_weight * torch.log(coverage_vector.transpose(0, 1).unsqueeze(2) + self.eps)

    # Transpose the attention vectors

    # transpose => (batch size, num encoder states, 1)

    enc_attn = F.softmax(enc_attn, dim=0).transpose(0, 1)

    

    # Calculate the context vectors 

    enc_context = torch.bmm(encoder_hidden.permute(1, 2, 0), enc_attn)

    # print(enc_context.shape,enc_context.squeeze(2).shape)

    

    # Update the combined vector with the context vectors 

    combined[:, offset:offset+enc_total_size] = enc_context.squeeze(2)

    offset += enc_total_size

    enc_attn = enc_attn.squeeze(2)

    

    # Set the decoder attention vectors

    if decoder_states is not None and len(decoder_states) > 0:

      dec_attn = self.dec_bilinear(hidden.expand_as(decoder_states).contiguous(),

                                      decoder_states)

      dec_attn = F.softmax(dec_attn, dim=0).transpose(0, 1)

      dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)

      combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)

      offset += self.hidden_size

    

    # Prepare the data to apply the pointer generator

    out_embed = combined

    logits = self.out(out_embed)  # (batch size, vocab size)



    # Distribute probabilities between generator and pointer

    prob_ptr = torch.sigmoid(self.ptr(combined))  # (batch size, 1)

    prob_gen = 1 - prob_ptr

    # add generator probabilities to output

    gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities

    output = prob_gen * gen_output

    # Dfine the padding dimension

    pad_dim = (0, ext_vocab_size - output.size(1))

    # Pad the output tensir

    output=F.pad(output, pad_dim, 'constant')



    # add pointer probabilities to output

    ptr_output = enc_attn

    encoder_word_idx_l = encoder_word_idx.long()

    # Calculate the output tensor    

    try:

        output.scatter_add_(1, encoder_word_idx_l, prob_ptr * ptr_output)

    except:

        prob_po = prob_ptr * ptr_output 

        print(output.shape,encoder_word_idx_l.shape,prob_ptr.shape, ptr_output.shape, prob_po.shape)

        print(output)

        print(encoder_word_idx_l)

        print(prob_po)

        output.scatter_add_(1, encoder_word_idx_l, prob_po)

        

    # Apply the log in the output

    output = torch.log(output + self.eps)



    return output, hidden, enc_attn, prob_ptr 

def get_coverage_vector(enc_attn_weights):

    """Combine the past attention weights into one vector"""

    coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)

    

    return coverage_vector  

def get_next_batch(data, p, vocab, i, batch_size, device):

    ''' Generate and return the next batch of the data during training

        Input:

        - data: list, input data to the model

        - p: a class Parameters object, model and training parameters

        - vocab: a class Vocab object, vocabulary of the data

        - i: integer, index or iterator

        - batch_size: integer, batch size

        - device: string, where to train the model, cpu or gpu 

    '''

    #Create a copy of the vocabulary

    vocab_ext=deepcopy(vocab)



    #Get the next batch

    try:

        data_dict=data[i:i+batch_size]

    except:

        data_dict=data[i:len(data)]

    # Create a batch from an extended cocabulary

    data_batch = sort_batch_by_len(data_dict,p,vocab_ext)

    # Create an extended cocabulary

    for word in data_dict['x']:

        vocab_ext.add_words(word)

            

    # Create a batch from an extended cocabulary

    data_batch_extra=sort_batch_by_len(data_dict,p,vocab_ext)

    #Create tha inputs in the extended version        

    x_extra=torch.tensor(data_batch_extra['x']).to(device)

    

    # Transform the batch to tensors

    x, x_len = torch.tensor(data_batch['x']).to(device), torch.tensor(data_batch['x_len']).to(device)

    y, y_len = torch.tensor(data_batch['y']).to(device), torch.tensor(data_batch['y_len']).to(device)



    return x, x_len, y, y_len, x_extra, vocab_ext

def train(dataset,val_dataset,vocab,p,embedding_weights, learning_rate, num_epochs):

    ''' Run all the steps in the training phase

        Input:

        - dataset: Dataset object, training data

        - val_dataset: Dataset object, validation data

        - vocab: a class Vocab object, the vocabulary of the datasets

        - p: a class Parameters object, model and training parameters

        - embedding_weigths: tensor, the embedding vectors

        - learning_rate: float, learning rate parameter

        - num_epochs: integer, number of epochs of the training

    '''

    # Set some variables like eps, batch size and device

    eps = p.eps

    batch_size =p.batch_size

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    #Create an adapter between encoder hidden state to decoder hidden size 

    enc_dec_adapter = nn.Linear(p.hidden_size * 2, p.dec_hidden_size).to(DEVICE)

    #Create an embedding layer with pretrained weigths

    embedding = nn.Embedding(len(vocab), p.embed_size, padding_idx=vocab.PAD,

                             _weight=embedding_weights).to(DEVICE)

    

    # Do not train the embeddings

    embedding.weight.requires_grad=False

    #Create the encoder

    encoder = EncoderRNN(p.embed_size, p.hidden_size, p.enc_bidi,rnn_drop=p.enc_rnn_dropout).to(DEVICE)

    #Create the decoder

    decoder = DecoderRNN(len(vocab), p.embed_size, p.dec_hidden_size,

                                  enc_attn=p.enc_attn, dec_attn=p.dec_attn,

                                  pointer=p.pointer,

                                  in_drop=p.dec_in_dropout, rnn_drop=p.dec_rnn_dropout,

                                  out_drop=p.dec_out_dropout, enc_hidden_size=p.hidden_size * 2,

                                  device=DEVICE, epsilon=p.eps).to(DEVICE)

    

    # If the model components have been training, we restore them from a previous save

    if(os.path.exists(p.encoder_weights_path) and p.resume_train):

        encoder.load_state_dict(torch.load(p.encoder_weights_path,map_location=torch.device(DEVICE)))

    if(os.path.exists(p.decoder_weights_path) and p.resume_train):

        decoder.load_state_dict(torch.load(p.decoder_weights_path,map_location=torch.device(DEVICE)))

    if(os.path.exists(p.encoder_decoder_adapter_weights_path) and p.resume_train):   

        enc_dec_adapter.load_state_dict(torch.load(p.encoder_decoder_adapter_weights_path,map_location=torch.device(DEVICE)))

    

    # Create a Dataset class containing the training data

    cnn_data=MyDataset([pair[0] for pair in dataset.pairs],[pair[1] for pair in dataset.pairs],vocab)

    

    # Create a Dataset class containing the validation data

    #CHECK IF CREATING A VOCAB FOR VALIDATION IS RIGHT

    val_data=MyDataset([pair[0] for pair in val_dataset.pairs],[pair[1] for pair in val_dataset.pairs],vocab)

    # print(cnn_data[:3]['x_len'])

    

    # DEfine the loss function

    criterion = nn.NLLLoss(ignore_index=vocab.PAD)

    # Define the optimizers for the encoder, decoder and the adapter

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    adapter_optimizer=optim.Adam([{'params':enc_dec_adapter.parameters()}], lr=learning_rate)

    # Record the losses

    losses=[]

    val_losses=[]

    #Load the losses from previous trainings

    if(os.path.exists(p.losses_path) and p.resume_train):

      with open(p.losses_path,'rb') as f:

        val_losses=pickle.load(f)

        

    #Run training for num_epochs

    for _e in range(num_epochs):

        i=0

        #Create a progress bar 

        print('\nEpoch: %d/%d' % (_e + 1, num_epochs))

        kbar = pkbar.Kbar(target=len(cnn_data), width=8)

        #for every batch in the training data 

        while i<len(cnn_data):

            

            # Reset the gradients for the forward phase

            encoder_optimizer.zero_grad()

            decoder_optimizer.zero_grad()

            adapter_optimizer.zero_grad()



            # Extract the data for the next batch

            x, x_len, y, y_len, x_extra, vocab_ext = get_next_batch(cnn_data, p, vocab, i, batch_size, device=DEVICE)

    

            # Apply the embedding layer in the encoder

            encoder_embedded = embedding(x)

            # Create the init hidden state of the encoder

            encoder_hidden=encoder.init_hidden(x.size(0), DEVICE)

            # Forward pass in the encoder

            encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,x_len)

            #Create the init input to the encoder

            decoder_input = torch.tensor([vocab.SOS] * x.size(0), device=DEVICE)

            # Adapt the encoder hidden to the encoder hidden size

            decoder_hidden = enc_dec_adapter(encoder_hidden)

            

            decoder_states = []

            enc_attn_weights = []

            loss=0

            # For every token in the target

            for di in range(y.size(1)):

                #Apply the embedding layer to the decoder input

                decoder_embedded = embedding(decoder_input)

                # If activation of encoder attention is on 

                if enc_attn_weights:

                    coverage_vector = get_coverage_vector(enc_attn_weights)

                else:

                    coverage_vector = None

                    

                #Forward pass to the decoder

                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,

                            torch.cat(decoder_states) if decoder_states else None, coverage_vector,

                            encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab_ext))  

                #Move the tensors to the device

                decoder_output.to(DEVICE)

                decoder_hidden.to(DEVICE)

                dec_enc_attn.to(DEVICE)

                dec_prob_ptr.to(DEVICE)

                

                #Save the decoder hidden state

                decoder_states.append(decoder_hidden)

                #Calculate the probability distribution of the decoder outputs

                prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output

                #Get the largest element 

                _, top_idx = decoder_output.data.topk(1)

                # Set the current target word to our goal

                gold_standard = y[:,di]

                # Apply the loss function

                nll_loss= criterion(decoder_output, gold_standard)    

                loss+=nll_loss

                

                #Set the decoder input to the target word or token 

                decoder_input = y[:,di]

                #Calculate the coverage loss

                if (coverage_vector is not None and criterion): 

                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size #* cover_loss            

                    loss+=coverage_loss

                    

                #Store the attention weights

                enc_attn_weights.append(dec_enc_attn.unsqueeze(0)) 

                

            #Apply the backward to get the loss

            loss.backward()

            # Clipping the weights in the encoder, decoder and the adapter

            clip_grad_norm_(encoder.parameters(), 1)

            clip_grad_norm_(decoder.parameters(), 1)

            clip_grad_norm_(enc_dec_adapter.parameters(), 1)

            # Update the parameters

            encoder_optimizer.step()

            decoder_optimizer.step()

            adapter_optimizer.step() 

            #Print the progress bar

            if i%(p.print_every*batch_size)==0:

                kbar.update(i, values=[("loss", loss.data.item())])

            # Get the next batch

            i+=batch_size

            

        # Calculate the final loss on the training    

        loss=loss.data.item()/x.size(0)

        kbar.add(1, values=[("loss", loss)])

        

        #Repeat the process on the validation dataset

        kbar2 = pkbar.Kbar(target=len(val_data), width=8)

        

        # calculating validation loss

        val_loss=0

        i=0

        while(i<len(val_data)):

            # Reset the gradients for the forward phase

            encoder_optimizer.zero_grad()

            decoder_optimizer.zero_grad()

            adapter_optimizer.zero_grad()

            # Get the next batch of the validation data

            x, x_len, y, y_len, x_extra, vocab_ext = get_next_batch(val_data, p, vocab, i, batch_size, device=DEVICE)

            # Forward pass of the encoder

            encoder_embedded = embedding(x)

            encoder_hidden=encoder.init_hidden(x.size(0), device=DEVICE)

            encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,x_len)

            decoder_input = torch.tensor([vocab.SOS] * x.size(0), device=DEVICE)

            decoder_hidden = enc_dec_adapter(encoder_hidden)

            

            decoder_states = []

            enc_attn_weights = []

            # For every word in the output

            for di in range(y.size(1)):

                try:

                    #Get the embedding vector of the input to the decoder

                    decoder_embedded = embedding(decoder_input)

                except:

                    print('Dec input: ',decoder_input.shape,' x:', x.shape,' x_len:',x_len.shape, ' Vocab:', 

                          vocab.embeddings.shape,' Vocab Ext:', vocab_ext.embeddings.shape)

                    decoder_embedded = embedding(decoder_input)

                # Generate the coverage vectors if neccessary

                if enc_attn_weights:

                    coverage_vector = get_coverage_vector(enc_attn_weights)

                else:

                    coverage_vector = None

                # Fordward pass ti the decoder

                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,

                            torch.cat(decoder_states) if decoder_states else None, coverage_vector,

                            encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab_ext))  

                # Move the tensors to the device, gou or cpu

                decoder_output.to(DEVICE)

                decoder_hidden.to(DEVICE)

                dec_enc_attn.to(DEVICE)

                dec_prob_ptr.to(DEVICE)

                # Stores the hidden states of the decoder

                decoder_states.append(decoder_hidden)      

                prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output

                # Get the output token with the highest probability

                _, top_idx = decoder_output.data.topk(1)

                gold_standard = y[:,di]

                # Apply the loss function

                nll_loss= criterion(decoder_output, gold_standard)    

                val_loss+=nll_loss.data.item()

                

                # Set the decoder input to the last output from the decoder

                decoder_input = top_idx.view(-1) #y[:,di]

                # update the coverage vector

                if (coverage_vector is not None and criterion): #and cover_loss > 0:

                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size #* cover_loss            

                    val_loss+=coverage_loss.data.item()

                # Collect the attention weights in the step    

                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))  

            # Print the progress

            if i%(p.print_every*batch_size)==0:

                kbar2.update(i, values=[("Val loss", val_loss)])



            i+=batch_size

            

        #Calculate the validation loss

        avg_val_loss=val_loss/len(val_data)        

        #print('training loss:{}'.format(loss),'validation loss:{}'.format(avg_val_loss))

        kbar2.add(1, values=[("Train loss", loss), ("Val loss", val_loss), ("Avg Val loss", avg_val_loss)])

        

        # Save the mnodel and results to disk

        if(len(val_losses)>0 and avg_val_loss<min(val_losses)):

            torch.save(encoder.state_dict(), p.encoder_weights_path)

            torch.save(decoder.state_dict(), p.decoder_weights_path)

            torch.save(enc_dec_adapter.state_dict(), p.encoder_decoder_adapter_weights_path)

            # torch.save(embedding.state_dict(), '/home/svu/e0401988/NLP/summarization/embedding_sum.pt')

        val_losses.append(avg_val_loss) 

    

    with open(p.losses_path,'wb') as f:

        pickle.dump(val_losses,f) 

def predict(sent,vocab,p,batch_size=1):

    ''' Function to predict the summary of the source text sentence

        Input:

        - sent: string, text to summarize

        - vocab: a class Vocab object, vocabulary of the texts

        - p: a class Parameters object, model parameters

        - batch_size: integer, batch size of the data to predict

    '''

    eps=p.eps

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a tensor with the embedding vectors

    embedding_weights = torch.from_numpy(vocab.embeddings).to(DEVICE)

    # Create the layer to transform inputs between encoder and decoder

    enc_dec_adapter = nn.Linear(p.hidden_size * 2, p.dec_hidden_size).to(DEVICE)

    # Create the embedding layer using the embedding vectors

    embedding = nn.Embedding(len(vocab), p.embed_size, padding_idx=vocab.PAD,_weight=embedding_weights).to(DEVICE)

    # Create the encoder and the decoder

    encoder = EncoderRNN(p.embed_size, p.hidden_size, p.enc_bidi,rnn_drop=p.enc_rnn_dropout).to(DEVICE)

    decoder = DecoderRNN(len(vocab), p.embed_size, p.dec_hidden_size,

                                  enc_attn=p.enc_attn, dec_attn=p.dec_attn,

                                  pointer=p.pointer,

                                  in_drop=p.dec_in_dropout, rnn_drop=p.dec_rnn_dropout,

                                  out_drop=p.dec_out_dropout, enc_hidden_size=p.hidden_size * 2,

                                  device=DEVICE).to(DEVICE) 

    # Tokenize the input text 

    sent_vec=[vocab[word] for word in sent.split()]

    vocab_ext=deepcopy(vocab)

    # Extend the vocabulary

    for word in sent.split():

      vocab_ext.add_words(word)

    # Tokenize the text with the extended  vocabulary

    sent_vec_extra=[vocab_ext[word] for word in sent.split()] 

    # Pad the text if neccessary

    if(len(sent_vec_extra)<p.max_src_len):

        pad_dim = (0, p.max_src_len-len(sent_vec_extra))

        #sent_vec_extra=F.pad(sent_vec_extra, pad_dim , 'constant')

        sent_vec_extra_tensor=F.pad(torch.tensor(sent_vec_extra), pad_dim , 'constant')

    else:

        sent_vec_extra_tensor=torch.tensor(sent_vec_extra)

        

    # Pad the text if neccessary

    if(len(sent_vec)<p.max_src_len):

        pad_dim = (0, p.max_src_len-len(sent_vec))

        #sent_vec=F.pad(sent_vec, pad_dim, 'constant')

        sent_vec_tensor=F.pad(torch.tensor(sent_vec), pad_dim, 'constant')

    else:

        sent_vec_tensor=torch.tensor(sent_vec)

        

    # Load the encoder model from file

    if(os.path.exists(p.encoder_weights_path)):

        encoder.load_state_dict(torch.load(p.encoder_weights_path,map_location=torch.device(DEVICE)))

    # Load the decoder model from file

    if(os.path.exists(p.decoder_weights_path)):

        decoder.load_state_dict(torch.load(p.decoder_weights_path,map_location=torch.device(DEVICE)))

    # Load the encoder -decoder adapter component from file

    if(os.path.exists(p.encoder_decoder_adapter_weights_path)):    

        enc_dec_adapter.load_state_dict(torch.load(p.encoder_decoder_adapter_weights_path,map_location=torch.device(DEVICE)))



    x=sent_vec_tensor.view(1,-1).to(DEVICE)

    x_extra=sent_vec_extra_tensor.view(1,-1).to(DEVICE)

    # Apply the embedding layer

    encoder_embedded = embedding(x)

    # Initiaize the hidden state of the encoder

    encoder_hidden=encoder.init_hidden(x.size(0), DEVICE)

    # Fordward pass to the encoder

    encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,

                                             torch.tensor(p.max_src_len).view(1).to(DEVICE))

    # Initialize the decoder input to SOS tokens

    decoder_input = torch.tensor([vocab.SOS] * batch_size, device=DEVICE)

    decoder_hidden = enc_dec_adapter(encoder_hidden)

    

    #Initialize the hidden states and weights of some variables

    decoder_states = []

    enc_attn_weights = []

    output=[]

    # For every word in the output sequence 

    for di in range(p.max_tgt_len):

        # Apply the embedding layer of the decoder

        decoder_embedded = embedding(decoder_input)

        # Get the coverage vector

        if enc_attn_weights:

            coverage_vector = get_coverage_vector(enc_attn_weights)

        else:

            coverage_vector = None

        # fordward pass to the decoder

        decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,

                    torch.cat(decoder_states).to(DEVICE) if decoder_states else None, coverage_vector,

                    encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab_ext))  

        # Move the tensor to the device

        decoder_output.to(DEVICE)

        decoder_hidden.to(DEVICE)

        dec_enc_attn.to(DEVICE)

        dec_prob_ptr.to(DEVICE)

        # Store the hidden state of the decoder

        decoder_states.append(decoder_hidden)

        # Generate the probability distribution

        prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output

        # Get the element in the decoder output with the highest probability (the best output)

        _, top_idx = decoder_output.data.topk(1)

        # Store the output (word) to the output text

        output.append(top_idx.squeeze().data.item())

        # Store the encoder attention weights

        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))

        # Set the decoder input in the next iter

        decoder_input = top_idx.view(-1)

    # Transform the outputs (words) to a list of text or words

    output=[vocab_ext[idx] for idx in output]    

    return output 
def prediction(sent,vocab,embedding, encoder, enc_dec_adapter, decoder, device, p,batch_size=1):

    ''' Function to predict the summary of the source text sentence

        Input:

        - sent: string, text to summarize

        - vocab: a class Vocab object, vocabulary of the texts

        - p: a class Parameters object, model parameters

        - batch_size: integer, batch size of the data to predict

    '''

    eps=p.eps

    # Tokenize the input text 

    sent_vec=[vocab[word] for word in sent.split()]

    vocab_ext=deepcopy(vocab)

    # Extend the vocabulary

    for word in sent.split():

      vocab_ext.add_words(word)

    # Tokenize the text with the extended  vocabulary

    sent_vec_extra=[vocab_ext[word] for word in sent.split()] 

    # Pad the text if neccessary

    if(len(sent_vec_extra)<p.max_src_len):

        pad_dim = (0, p.max_src_len-len(sent_vec_extra))

        #sent_vec_extra=F.pad(sent_vec_extra, pad_dim , 'constant')

        sent_vec_extra_tensor=F.pad(torch.tensor(sent_vec_extra), pad_dim , 'constant')

    else:

        sent_vec_extra_tensor=torch.tensor(sent_vec_extra)

        

    # Pad the text if neccessary

    if(len(sent_vec)<p.max_src_len):

        pad_dim = (0, p.max_src_len-len(sent_vec))

        #sent_vec=F.pad(sent_vec, pad_dim, 'constant')

        sent_vec_tensor=F.pad(torch.tensor(sent_vec), pad_dim, 'constant')

    else:

        sent_vec_tensor=torch.tensor(sent_vec)

        

    x=sent_vec_tensor.view(1,-1).to(device)

    x_extra=sent_vec_extra_tensor.view(1,-1).to(device)

    # Apply the embedding layer

    encoder_embedded = embedding(x)

    # Initiaize the hidden state of the encoder

    encoder_hidden=encoder.init_hidden(x.size(0), device)

    # Fordward pass to the encoder

    encoder_outputs, encoder_hidden =encoder(encoder_embedded,encoder_hidden,

                                             torch.tensor(p.max_src_len).view(1).to(device))

    # Initialize the decoder input to SOS tokens

    decoder_input = torch.tensor([vocab.SOS] * batch_size, device=device)

    decoder_hidden = enc_dec_adapter(encoder_hidden)

    

    #Initialize the hidden states and weights of some variables

    decoder_states = []

    enc_attn_weights = []

    output=[]

    # For every word in the output sequence 

    for di in range(p.max_tgt_len):

        # Apply the embedding layer of the decoder

        decoder_embedded = embedding(decoder_input)

        # Get the coverage vector

        if enc_attn_weights:

            coverage_vector = get_coverage_vector(enc_attn_weights)

        else:

            coverage_vector = None

            

        # fordward pass to the decoder

        decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = decoder(decoder_embedded, decoder_hidden, encoder_outputs,

                    torch.cat(decoder_states).to(device) if decoder_states else None, coverage_vector,

                    encoder_word_idx=x_extra,log_prob=True,ext_vocab_size=len(vocab_ext))  

        # Move the tensor to the device

        decoder_output.to(device)

        decoder_hidden.to(device)

        dec_enc_attn.to(device)

        dec_prob_ptr.to(device)

        # Store the hidden state of the decoder

        decoder_states.append(decoder_hidden)

        # Generate the probability distribution

        prob_distribution = torch.exp(decoder_output)# if log_prob else decoder_output

        # Get the element in the decoder output with the highest probability (the best output)

        _, top_idx = decoder_output.data.topk(1)

        # Store the output (word) to the output text

        output.append(top_idx.squeeze().data.item())

        # Store the encoder attention weights

        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))

        # Set the decoder input in the next iter

        decoder_input = top_idx.view(-1)

        

    # Transform the outputs (words) to a list of text or words

    output=[vocab_ext[idx] for idx in output]    

    return output 
def eval_metrics(preds, targets, avg=True):

    ''' Evaluate the ROUGE metrics ROUGE-2 and ROUGE-L for every pair predicted summary - target summary

    

        Input:

           - preds: list of strings, predicted summaries

           - targets: list of string, target summaries

        Output:

            - rouge2_f_metric: list of float, the Rouge-2 fscore for every predicted summary

            - rougel_f_metric: list of float, the Rouge-L fscore for every predicted summary

    '''

    #Lets calculate the rouge metrics for every document

    rouge = Rouge()

    scores = rouge.get_scores(preds, targets, avg)

    # Create the output variables

    if avg:

        rouge2_f_metric = scores['rouge-2']['f']

        rouge2_p_metric = scores['rouge-2']['p']

        rouge2_r_metric = scores['rouge-2']['r']

        rougel_f_metric = scores['rouge-l']['f']

        rougel_p_metric = scores['rouge-l']['p']

        rougel_r_metric = scores['rouge-l']['r']

    else:

        rouge2_f_metric = [score['rouge-2']['f'] for score in scores]

        rouge2_p_metric = [score['rouge-2']['p'] for score in scores]

        rouge2_r_metric = [score['rouge-2']['r'] for score in scores]

        rougel_f_metric = [score['rouge-l']['f'] for score in scores]

        rougel_p_metric = [score['rouge-l']['p'] for score in scores]

        rougel_r_metric = [score['rouge-l']['r'] for score in scores]

    

    return rouge2_f_metric, rouge2_p_metric, rouge2_r_metric, rougel_f_metric, rougel_p_metric, rougel_r_metric



def save_to_df(text, labeled_summaries, predicted_summaries, r2_f, r2_p, r2_r, rl_f, rl_p, rl_r):

    ''' Stores the metric results into a pandas dataframe'''

    results = pd.DataFrame(columns=['text', 'summary','pred_summary','rouge2-f','rouge2-p','rouge2-r','rougel-f', 'rougel-p', 'rougel-r'])

    results['text'] = text

    results['summary'] = labeled_summaries

    results['pred_summary'] = predicted_summaries

    results['rouge2-f'] = r2_f

    results['rouge2-p'] = r2_p

    results['rouge2-r'] = r2_r

    results['rougel-f'] = rl_f

    results['rougel-p'] = rl_p

    results['rougel-r'] = rl_r



    return results
def get_predictions(x_test, vocab, params, print_every=20):

    ''' Generate the predicted summaries of the source texts on x_test

        Input:

        - x_test: list of strings, the source texts

        - vocab: a Vocab Class object, vocabulary of the texts

        - params: a Parameters object, parameter of the model

        - print_every: integer, print progress every print_every iterations

    '''

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a tensor with the embedding vectors

    embedding_weights = torch.from_numpy(vocab.embeddings).to(DEVICE)

    # Create the layer to transform inputs between encoder and decoder

    enc_dec_adapter = nn.Linear(params.hidden_size * 2, params.dec_hidden_size).to(DEVICE)

    # Create the embedding layer using the embedding vectors

    embedding = nn.Embedding(len(vocab), params.embed_size, padding_idx=vocab.PAD,_weight=embedding_weights).to(DEVICE)

    # Create the encoder and the decoder

    encoder = EncoderRNN(params.embed_size,params.hidden_size, params.enc_bidi,rnn_drop=params.enc_rnn_dropout).to(DEVICE)

    decoder = DecoderRNN(len(vocab), params.embed_size, params.dec_hidden_size,

                                  enc_attn=params.enc_attn, dec_attn=params.dec_attn,

                                  pointer=params.pointer,

                                  in_drop=params.dec_in_dropout, rnn_drop=params.dec_rnn_dropout,

                                  out_drop=params.dec_out_dropout, enc_hidden_size=params.hidden_size * 2,

                                  device=DEVICE).to(DEVICE) 



    # Load the encoder model from file

    if(os.path.exists(params.encoder_weights_path)):

        encoder.load_state_dict(torch.load(params.encoder_weights_path,map_location=torch.device(DEVICE)))

    # Load the decoder model from file

    if(os.path.exists(params.decoder_weights_path)):

        decoder.load_state_dict(torch.load(params.decoder_weights_path,map_location=torch.device(DEVICE)))

    # Load the encoder -decoder adapter component from file

    if(os.path.exists(params.encoder_decoder_adapter_weights_path)):    

        enc_dec_adapter.load_state_dict(torch.load(params.encoder_decoder_adapter_weights_path,map_location=torch.device(DEVICE)))



        

    predicted_summaries = []

    # Set a progress bar

    kbar = pkbar.Kbar(target=len(x_test), width=8)

    # For every text in the validation dataset

    for i,doc in enumerate(x_test):

        # Predict the summary for the document

        pred_summ = prediction(doc,vocab,embedding,encoder,enc_dec_adapter,decoder,DEVICE,params,batch_size=1)

        #pred_summ = predict(doc,vocab,params,batch_size=1)

        predicted_summaries.append(' '.join(pred_summ))

        #Show the progress

        if i%print_every==0:

            kbar.update(i)

            

    # Set the labeled summaries as the y_test variable, column summary of our dataset

    return predicted_summaries

def generate_predictions(x_test, vocab, params, print_every=20):

    ''' Generate the predicted summaries of the source texts on x_test

        Input:

        - x_test: list of strings, the source texts

        - vocab: a Vocab Class object, vocabulary of the texts

        - params: a Parameters object, parameter of the model

        - print_every: integer, print progress every print_every iterations

    '''



    predicted_summaries = []

    # Set a progress bar

    kbar = pkbar.Kbar(target=len(x_test), width=8)

    # For every text in the validation dataset

    for i,doc in enumerate(x_test):

        # Predict the summary for the document

        pred_summ = predict(doc,vocab,params,batch_size=1)

        predicted_summaries.append(' '.join(pred_summ))

        #Show the progress

        if i%print_every==0:

            kbar.update(i)

            

    # Set the labeled summaries as the y_test variable, column summary of our dataset

    return predicted_summaries

# Create an object with the model and training parameters

params = Parameters()
# Load the training dataset using the simple tokenizer

dataset = Dataset(params.data_path, simple_tokenizer, params.max_src_len, params.max_tgt_len, max_rows=64000,

                        truncate_src=True, truncate_tgt=True)

# Load the validation dataset using the simple tokenizer

valid_dataset = Dataset(params.val_data_path, simple_tokenizer, params.max_src_len, params.max_tgt_len, max_rows= 3200,

                        truncate_src=True, truncate_tgt=True)

#Show the length to check the loadings

print(dataset.src_len, valid_dataset.src_len,dataset.tgt_len, valid_dataset.tgt_len)
# convert the embeddings to a tensor

vocab = dataset.build_vocab(params.vocab_min_frequency, embed_file=params.embed_file)

vocab.save_to_file('vocab_train.pkl')

# convert the embeddings to a tensor

embedding_weights = torch.from_numpy(vocab.embeddings)
# Load the vocab from file

#vocab= load_vocab('/kaggle/input/pointer-trained-model/vocab_train.pkl')

#print(vocab.embeddings.shape, len(vocab.index2word))

#embedding_weights = torch.from_numpy(vocab.embeddings)
train(dataset,valid_dataset,vocab, params, embedding_weights,learning_rate=0.001,num_epochs = 10)
test_dataset = Dataset(params.test_data_path, simple_tokenizer, params.max_src_len, params.max_tgt_len, max_rows= 3200,

                        truncate_src=True, truncate_tgt=True)

# Prepare the validation dataset to be used in the evaluation

print('Length Test Dataset:', len(test_dataset.pairs))

x_test = [' '.join(pair[0]) for pair in test_dataset.pairs]

y_test = [' '.join(pair[1]) for pair in test_dataset.pairs]

# Predict the summaries

#preds = generate_predictions(x_test, vocab, params, print_every=10)

preds = get_predictions(x_test, vocab, params, print_every=100)

# Calculate the Rouge-2 and Rouge-L metrics for the validation dataset

r2_f, r2_p, r2_r, rl_f, rl_p, rl_r = eval_metrics(preds, y_test, False)

print('\nMean Rouge-2 FScore: ',np.mean(r2_f), 'Mean Rouge-L FScore: ',np.mean(rl_f))

# Store the evaluation results to a CSV file

test_results = save_to_df(x_test, y_test, preds, r2_f, r2_p, r2_r, rl_f, rl_p, rl_r)

test_results.to_csv('test_results_pointer_gen.csv', index=False)

test_results.head(5)