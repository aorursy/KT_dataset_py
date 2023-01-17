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
dataset=pd.read_csv('/kaggle/input/news-summary/news_summary.csv', encoding='cp437')
dataset=dataset[['ctext','text']] #add headlines and see if it makes a difference
dataset.head(1)
dataset['text'][0]
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
EOS_token = 1
SOS_token=0

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        sentence = removeCont(sentence)
        for word in sentence.split(' '):
            if word not in stop_words:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

contractions = {"ain't": "is not",
                "aren't": "are not",
                "can't": "cannot",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'll": "he will",
                "he's": "he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how is",
                "I'd": "I would",
                "I'd've": "I would have",
                "I'll": "I will", 
                "I'll've": "I will have",
                "I'm": "I am", 
                "I've": "I have", 
                "i'd": "i would",
                "i'd've": "i would have", 
                "i'll": "i will",  
                "i'll've": "i will have",
                "i'm": "i am", 
                "i've": "i have", 
                "isn't": "is not", 
                "it'd": "it would",
                "it'd've": "it would have", 
                "it'll": "it will", 
                "it'll've": "it will have",
                "it's": "it is", 
                "let's": "let us", 
                "ma'am": "madam",
                "mayn't": "may not", 
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have", 
                "must've": "must have",
                "mustn't": "must not", 
                "mustn't've": "must not have", 
                "needn't": "need not", 
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not", 
                "oughtn't've": "ought not have", 
                "shan't": "shall not", 
                "sha'n't": "shall not", 
                "shan't've": "shall not have",
                "she'd": "she would", 
                "she'd've": "she would have", 
                "she'll": "she will", 
                "she'll've": "she will have", 
                "she's": "she is",
                "should've": "should have", 
                "shouldn't": "should not", 
                "shouldn't've": "should not have", 
                "so've": "so have",
                "so's": "so as",
                "this's": "this is",
                "that'd": "that would",
                "that'd've": "that would have", 
                "that's": "that is", 
                "there'd": "there would",
                "there'd've": "there would have", 
                "there's": "there is", 
                "here's": "here is",
                "they'd": "they would", 
                "they'd've": "they would have",
                "they'll": "they will", 
                "they'll've": "they will have", 
                "they're": "they are", 
                "they've": "they have", 
                "to've": "to have",
                "wasn't": "was not", 
                "we'd": "we would", 
                "we'd've": "we would have", 
                "we'll": "we will", 
                "we'll've": "we will have", 
                "we're": "we are",
                "we've": "we have", 
                "weren't": "were not", 
                "what'll": "what will", 
                "what'll've": "what will have", 
                "what're": "what are",
                "what's": "what is", 
                "what've": "what have", 
                "when's": "when is", 
                "when've": "when have", 
                "where'd": "where did", 
                "where's": "where is",
                "where've": "where have", 
                "who'll": "who will", 
                "who'll've": "who will have", 
                "who's": "who is", 
                "who've": "who have",
                "why's": "why is", 
                "why've": "why have", 
                "will've": "will have", 
                "won't": "will not", 
                "won't've": "will not have",
                "would've": "would have", 
                "wouldn't": "would not", 
                "wouldn't've": "would not have", 
                "y'all": "you all",
                "y'all'd": "you all would",
                "y'all'd've": "you all would have",
                "y'all're": "you all are",
                "y'all've": "you all have",
                "you'd": "you would", 
                "you'd've": "you would have", 
                "you'll": "you will", 
                "you'll've": "you will have",
                "you're": "you are", 
                "you've": "you have",
                "%" :" percent"}
import re
def removeCont(text):
    if isinstance(text, float):
        return str(text)
    text=text.lower()
    text=text.split()
    for i in range(len(text)):
        if isinstance(text[i],float):
            text[i]=str(text)
        if text[i] in contractions:
            text[i]=contractions[text[i]]
    text=' '.join(text)
    text = text.replace("'s",'') # convert your's -> your
    text = re.sub(r'\(.*\)','',text) # remove (words)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations
    text = re.sub(r'\.',' . ',text)
    text = text.replace('.','')
    text=text.split()
    newtext=""
    for word in text:
        if word not in stop_words:
            newtext+=" "+word
    return newtext
removeCont("do...").split()
for i in range(len(dataset['ctext'])):
    dataset['ctext'][i]=removeCont(dataset['ctext'][i])
    dataset['text'][i]=removeCont(dataset['text'][i])
dataset['ctext'][100]
lang=Lang()
print(len(dataset['ctext']))
len(dataset['text'])
for i in range(len(dataset['text'])):
    if isinstance(dataset['text'][i],float) or isinstance(dataset['ctext'][i],float):
        continue
    lang.addSentence(dataset['text'][i])
    lang.addSentence(dataset['ctext'][i])
lang.word2index['daman']
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim
from tqdm import tqdm
from torchtext import data
embeddings_index={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()

print('Found %s word vectors in the GloVe library' % len(embeddings_index))
MAX_LENGTH = 100
EMBEDDING_DIM=100
matrix_len = len(lang.word2index)
weights_matrix = np.zeros((matrix_len, 100))
words_found = 0

for i, word in enumerate(lang.word2index):
    try: 
        weights_matrix[i] = embeddings_index[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM, ))
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings=len(lang.word2index)
    emb_layer = nn.Embedding(num_embeddings, EMBEDDING_DIM)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        num_embeddings=len(lang.word2index)
        #self.embedding = create_emb_layer(weights_matrix)
        self.embedding = nn.Embedding(num_embeddings, EMBEDDING_DIM)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        num_embeddings=len(lang.word2index)
        #self.embedding = create_emb_layer(weights_matrix)
        self.embedding = nn.Embedding(num_embeddings, EMBEDDING_DIM)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
def indexesFromSentence(sentence):
    #print(sentence)
    sent=sentence.split()
    sent.pop(0)
    l=[]
    c=0
    #print(sent)
    for word in sent:
        word=word.replace('.','')
        try:
            l.append(lang.word2index[word])
        except:
            c+=1
    return l

def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0].to_string().lower())
    target_tensor = tensorFromSentence(pair[1].to_string().lower())
    return (input_tensor, target_tensor)
teacher_forcing=0.5
sample_with_replacement = dataset.sample(n=1,replace=True)
print(sample_with_replacement['ctext'])
def getrandom():
    l = []
    sample_with_replacement = dataset.sample(n=1,replace=True)
    l.append(sample_with_replacement['ctext'])
    l.append(sample_with_replacement['text'])
    return l
def calcLoss(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size)
    loss = 0
    encoder_hidden=encoder.initHidden()
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden
    
    #Teacher forcing
    for i in range(target_length): #, decoder_attention  , encoder_outputs
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[i])
        decoder_input = target_tensor[i]
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
def train(encoder, decoder, n_iters, learning_rate=0.01):
    epoch_loss,epoch_acc=0,0
    plot_losses = []
    print_every=1000
    print_loss_total = 0  

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(getrandom())
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    encoder.train()
    decoder.train()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        #print(input_tensor)
        #break
        loss = calcLoss(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        #plot_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(' (%d %d%%) %.4f', iter, iter / n_iters * 100, print_loss_avg)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
        encoder.eval()
        decoder.eval()
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)

        for i in range(max_length):#, decoder_attention , encoder_outputs
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #decoder_attentions[i] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            #, decoder_attentions[:di + 1]
        return decoded_words
hidden_size = 100
encoder = Encoder(lang.n_words, hidden_size)
decoder=Decoder(hidden_size, lang.n_words)
train(encoder,decoder, 7500)