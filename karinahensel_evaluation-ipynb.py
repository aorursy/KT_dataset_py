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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import os
import math

""" Utility functions to load the data sets and preprocess the input """

TAG_INDICES = {'I-PER':0, 'B-PER':1, 'I-LOC':2, 'B-LOC':3, 'I-ORG':4, 'B-ORG':5, 'I-MISC':6, 'B-MISC':7, 'O':8}

def load_sentences(filepath):
    """
    Load sentences (separated by newlines) from dataset

    Parameters
    ----------
    filepath : str
        path to corpus file

    Returns
    -------
    List of sentences represented as dictionaries

    """
    
    sentences, tok, pos, chunk, ne = [], [], [], [], []

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line == ('-DOCSTART- -X- -X- O\n') or line == '\n':
               # Sentence as a sequence of tokens, POS, chunk and NE tags
                sentence = dict({'TOKENS' : [], 'POS' : [], 'CHUNK_TAG' : [], 'NE' : [], 'SEQ' : []})
                sentence['TOKENS'] = tok
                sentence['POS'] = pos
                sentence['CHUNK_TAG'] = chunk
                sentence['NE'] = ne
               
                # Once a sentence is processed append it to the list of sentences
                sentences.append(sentence)
               
                # Reset sentence information
                tok = []
                pos= []
                chunk = []
                ne = []
            else:
                l = line.split(' ')
               
                # Append info for next word
                tok.append(l[0])
                pos.append(l[1])
                chunk.append(l[2])
                ne.append(l[3].strip('\n'))
    
    return sentences
    
def read_conll_datasets(data_dir):
    data = {}
    for data_set in ["train","test","valid"]:
        data[data_set] = load_sentences("%s/%s.txt" % (data_dir,data_set))
    return data

def save_model(model, name, optimizer, loss):
    """
    Print evaluation of saved model

    Parameters
    ----------
    model : Model
        BiLSTM model loaded from file.
    name : String
        File name.
    """
    torch.save(model, name +'.pt')
    torch.save(model.state_dict(), name + '_2.pt')
    torch.save({
    'model': model.state_dict(),
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss.state_dict()
}, name + '_state_dict.pt')
    

def load_model(model_file, device):
    """
    Load a model from a file and print evaluation

    Parameters
    ----------
    model_file : String
        Path to model file.
    """
    m1 = torch.load(model_file, map_location=device)
    return m1

def prepare_emb(sent, tags, words_to_ix, tags_to_ix):
    w_idxs, tag_idxs = [], []
    for w, t in zip(sent, tags):
        if w.lower() in words_to_ix.keys():
            w_idxs.append(words_to_ix[w.lower()])
        else:
            w_idxs.append(words_to_ix['unk'])
                                      
        if t in tags_to_ix.keys():
            tag_idxs.append(tags_to_ix[t])
        else:
            tag_idxs.append(tags_to_ix['O'])
            
    return torch.tensor(w_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)

class Model(Module):
    def __init__(self, pretrained_embeddings, hidden_size, vocab_size, n_classes):
        super(Model, self).__init__()
        
        # Vocabulary size
        self.vocab_size = pretrained_embeddings.shape[0]
        # Embedding dimensionality
        self.embedding_size = pretrained_embeddings.shape[1]
        # Number of hidden units
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        
        # Dropout
        #self.dropout = Dropout(p=0.5, inplace=False)
        
        # Hidden layer (300, 20)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=2)
        
        # Final prediction layer
        self.hidden2tag = Linear(self.hidden_size, n_classes)#, bias=True)
    
    def forward(self, x):
        # Retrieve word embedding for input token
        emb = self.embedding(x)
        # Apply dropout
        #dropout = self.dropout(emb)
        # Hidden layer
        h, _ = self.lstm(emb.view(len(x), 1, -1))#self.lstm(emb.view(len(x), 1, -1))
        # Prediction
        pred = self.hidden2tag(h.view(len(x), -1))
        
        return F.log_softmax(pred, dim=1)
embeddings_file = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'
data_dir = '/kaggle/input/conll003-englishversion/'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load data
dataset = read_conll_datasets(data_dir)
#gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
glove2word2vec(glove_input_file=embeddings_file, word2vec_output_file="gensim_glove_vectors.txt")
gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format("./gensim_glove_vectors.txt", binary=False)
pretrained_embeds = gensim_embeds.vectors
# To convert words in the input to indices of the embeddings matrix:
word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}
    
# Hyperparameters
# Number of output classes (9)
n_classes = len(TAG_INDICES)
# Epochs
n_epochs = 2
report_every = 1
verbose = True
    
# Set up and initialize model
model = Model(pretrained_embeds, 100, len(word_to_idx), n_classes)
loss_function = NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

# Training loop
for e in range(n_epochs):
    total_loss, num_words = 0, 0
        
    for sent in dataset["train"][:10]:
            
            # (1) Set gradient to zero for new example: Set gradients to zero before pass
            model.zero_grad()
            
            # (2) Encode sentence and tag sequence as sequences of indices
            input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
            num_words += len(sent["TOKENS"])
            
            # (3) Predict tags (sentence by sentence)
            if len(input_sent) > 0:
                pred_scores = model(input_sent.to(device))
                
                # (4) Compute loss and do backward step
                loss = loss_function(pred_scores.to(device), gold_tags.to(device))
                loss.backward()
              
                # (5) Optimize parameter values
                optimizer.step()
          
                # (6) Accumulate loss
                total_loss += loss
    if ((e+1) % report_every) == 0:
            print('Epoch: %d, loss: %.4f' % (e, total_loss*100/num_words))
            
# Save the trained model
#save_model(model, 'm1_50e_dropout', optimizer, loss_function)
!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install pytorch_lightning
from pytorch_lightning.metrics.classification import F1
from sklearn.metrics import f1_score, accuracy_score

# To convert words in the input to indices of the embeddings matrix:
word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}

# Metrics
f1 = F1()

# Evaluation on the test set

# No dropout
model_nodropout = load_model('/kaggle/input/ner-trainedlstms/m1_50e_nodropout.pt', device)
correct, num_words, f1_scores, loss, num_sents = 0, 0, 0, 0, 0
with torch.no_grad():
    for sent in dataset["test"]:
        
        num_words += len(sent["TOKENS"])
        input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
        
        predicted, cor = 0.0, 0.0

        # Predict class with the highest probability
        if len(input_sent) > 0:
            num_sents += 1
            predicted = torch.argmax(model_nodropout(input_sent.to(device)), dim=1)
            
            correct += torch.sum(torch.eq(predicted.to(device),gold_tags.to(device)))
            f1_scores += f1_score(predicted.cpu(), gold_tags.cpu(), average='weighted')

print('Evaluation of the word-based LSTM without dropout:')
print('----------------------------------------------------------')
print('Test set accuracy: %.2f' % (100.0 * correct / num_words))
print('Test set f1-score: %.2f' % (100 * f1_scores / num_sents))
# Dropout
model_dropout = load_model('/kaggle/input/ner-trainedlstms/m1_50e_dropout.pt', device)
correct, num_words, f1_scores, num_sents = 0, 0, 0, 0
with torch.no_grad():
    for sent in dataset["test"]:
        
        num_words += len(sent["TOKENS"])
        input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
        
        predicted, cor = 0.0, 0.0

        # Predict class with the highest probability
        if len(input_sent) > 0:
            num_sents += 1
            predicted = torch.argmax(model_dropout(input_sent.to(device)), dim=1)
            
            correct += torch.sum(torch.eq(predicted.to(device),gold_tags.to(device)))
            f1_scores += f1_score(predicted.cpu(), gold_tags.cpu(), average='weighted')

print('Evaluation of the word-based LSTM with dropout:')
print('----------------------------------------------------------')
print('Test set accuracy: %.2f' % (100.0 * correct / num_words))
print('Test set f1-score: %.2f' % (100 * f1_scores / num_sents))