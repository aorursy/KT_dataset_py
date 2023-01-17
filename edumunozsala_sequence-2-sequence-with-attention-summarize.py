# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#Import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
import matplotlib.ticker as ticker

#Import libraries for text processing
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

stop_words = stopwords.words('english')

#Import some utils
from io import open
import unicodedata
import random
import pickle

#Import the pytorch libraries and modules
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

!pip install rouge
#Import library to calculate the evaluation metric
from rouge import Rouge
data_path = '/kaggle/input/cleaned-news-summary/cl_train_news_summary_more.csv'
valid_path = '/kaggle/input/cleaned-news-summary/cl_valid_news_summary_more.csv'
# Read the csv file
data = pd.read_csv(data_path,encoding='utf-8')
#Drop rows with duplicate values in the text column
data.drop_duplicates(subset=["text"],inplace=True)
#Drop rows with null values in the text variable
data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)
# we are using the text variable as the summary and the ctext as the source text
print('Drop null and duplicates, Total rows:', len(data))
# Rename the columns
data.columns = ['summary','text']
data.head()
# Read the csv file
valid_dataset = pd.read_csv(valid_path,encoding='utf-8', nrows=10000)
#Drop rows with duplicate values in the text column
valid_dataset.drop_duplicates(subset=["text"],inplace=True)
#Drop rows with null values in the text variable
valid_dataset.dropna(inplace=True)
valid_dataset.reset_index(drop=True,inplace=True)
# we are using the text variable as the summary and the ctext as the source text
print('Drop null and duplicates, Total rows:', len(valid_dataset))
# Rename the columns
valid_dataset.columns = ['summary','text']
valid_dataset.head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
def preprocess(text):
    ''' Function to clean the input text: convert to lowercase, expand the contractions, remove the stopwords,
        remove punctuations
    '''

    text = text.lower() # lowercase
    text = text.split() # convert have'nt -> have not
    
    for i in range(len(text)): # For every token or word in the text
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word] # Expand the contractions
            
    text = " ".join(text) # Rejoin the word to a sentence
    text = text.split() # Split the text into words
    newtext = []
    for word in text: # For every token or word in the text
        if word not in stop_words:
            newtext.append(word) #Include only the non stopwords
    text = " ".join(newtext)
    text = text.replace("'s",'') # Expand contractions, convert your's -> your
    text = re.sub(r'\(.*\)','',text) # remove (words)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations
    text = re.sub(r'\.',' . ',text)
    return text

data['summary'] = data['summary'].apply(lambda x:preprocess(x))
data['text'] = data['text'].apply(lambda x:preprocess(x))
valid_dataset['summary'] = data['summary'].apply(lambda x:preprocess(x))
valid_dataset['text'] = data['text'].apply(lambda x:preprocess(x))
data['summary'][20]
data['text'][20]
x = data['text']
y = data['summary']
print(x[50],y[50],sep='\n')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        ''' Add every word in a sentence to the vocabulary '''
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        ''' Add a word to the vocabulary'''
        if word not in self.word2index:
            #Include the word in the mapping from word to index
            self.word2index[word] = self.n_words
            #Set the count of ocurrencies of the word to 1
            self.word2count[word] = 1
            # Include the word in the indexes
            self.index2word[self.n_words] = word
            # Increment by 1 the number of words
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def save_to_file(self, filename):
        ''' Save the Vocab object to a file'''
        with open(filename,'wb') as f:
            pickle.dump(self,f) 

def load_vocab(filename):
    ''' Load a Vocab instance from a file'''
    with open(filename,'rb') as f:
        v = pickle.load(f)
    return v

def read_vocabs(text, summary, reverse=False):
    print("Reading lines...")
    
    # Split every line into pairs and normalize
    pairs = [[text[i],summary[i]] for i in range(len(text))]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Vocab(summary)
        output_lang = Vocab(text)
    else:
        input_lang = Vocab(text)
        output_lang = Vocab(summary)

    return input_lang, output_lang, pairs

def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_vocabs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
# Create the vocabularies of the inout and output data and return the data in pairs of (source text, summary)
input_lang, output_lang, pairs = prepare_data( x, y , False)
print(random.choice(pairs))
len_x_tr=[]
len_y_tr=[]
# For every pair Text, summary on the training dataset
for i in pairs:
    len_x_tr.append(len(i[0].split(' '))) # Get the count of words for the soure text
    len_y_tr.append(len(i[1].split(' '))) # Get the count of words for the summary
    
# 
x_test = valid_dataset['text'].values
y_test = valid_dataset['summary'].values

len_x_val=[]
len_y_val=[]
# For every pair Text, summary
for i in range(len(x_test)):
    len_x_val.append(len(x_test[i].split(' '))) # Get the count of words for the soure text
    len_y_val.append(len(y_test[i].split(' '))) # Get the count of words for the summary

print('Max Length of Texts: ', max(len_x_tr), 'Max Length of Summaries: ',max(len_x_val))
# Set the global variable MAX LENGTH
MAX_LENGTH = max(max(len_x_tr), max(len_y_tr), max(len_x_val), max(len_y_val), )+1
print(MAX_LENGTH)
class EncoderRNN(nn.Module):
    ''' Define an encoder in a seq2seq architecture'''
    def __init__(self, input_size, hidden_size):
        ''' Initialize tyhe encoder instance defining its parameters:
            Input:
                - input_size: the size of the vocabulary
                - hidden:size: size of the hidden layer
        '''
        super(EncoderRNN, self).__init__()
        # Set the hidden size
        self.hidden_size = hidden_size
        # Create the embedding layer of size (vocabulary length, hidden_size) 
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Create a GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        ''' Run a Forward pass of the encoder to return outputs
            Input:
                Input: a tensor element (integer) representing the next word in the sentence
                hidden: a tensor, the previous hidden state of the encoder
        '''
        # Get the embedding of the input
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        
        # Apply a forward step of the GRU returning the output features and
        # the hidden state of the actual time step
        output, hidden = self.gru(output, hidden)
        
        return output, hidden

    def initHidden(self):
        ''' Initialize the hidden state of the encoder, tensor of zeros'''
        return torch.zeros(1, 1, self.hidden_size, device=device)
class AttnDecoderRNN(nn.Module):
    ''' Define a decoder with atention in a seq2seq architecture'''
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        ''' Initialize the decoder instance defining its parameters:
            Input:
                - hidden_size:size: size of the hidden layer (Hyperparameter)
                - output_size: the size of the vocabulary of the output summary
                - dropout_p: dropout probability to apply
                - max_length: max length (number of words) of an output or summary
        '''

        super(AttnDecoderRNN, self).__init__()
        # Set parameters of the decoder
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        #Create an embedding layer for the input (output vocabulary, hidden size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Create some linear layers to build the attention mechanism
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # A dropout layer
        self.dropout = nn.Dropout(self.dropout_p)
        # A GRU layer
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # A Fully-connected layer
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        ''' Run a Forward pass of the decoder to return outputs
            Input:
                Input: a tensor element (integer) representing the previous output of the decoder
                hidden: a tensor, the previous hidden state of the decoder
                Encoder outputs: a tensor, outputs of the encoder
        '''
        
        #Get the embedding representation of the input
        embedded = self.embedding(input).view(1, 1, -1)
        # Apply dropout 
        embedded = self.dropout(embedded)
        #Calculate the attention weights of the attention mechanism using the encoder states
        #in previous time steps
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        #Calculate the context vectors fo the attention mechanism using the attention weights
        # and the encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        # Apply a forward pass to the GRU layer of the decider using the output from the attention
        # as input and the hidden state
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        # return the output features, the hidden state and the attention weights
        return output, hidden, attn_weights

    def initHidden(self):
        ''' Initialize the hidden state of the encoder, tensor of zeros'''
        return torch.zeros(1, 1, self.hidden_size, device=device)
def indexesFromSentence(lang, sentence):
    ''' Transform a sentence in string format to a list of indexes or integers.
            The model need to be feeded with numbers, not characters
            Input:
                - sentence: a string
            Output:
                - a list of integers, the representation of the sentence in the vector space.
    '''
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    ''' Transform a sentence in string format to tensor of indexes or integers.
            Out pytorch model work with tensor objects
            Input:
                - sentence: a string
            Output:
                - a tensor of integers, the representation of the sentence in the vector space.
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    ''' Convert a pair of text data (source text, summary) to tensors
        Input:
        - pair: tuple of strings, the source text and its summary
        Output:
        - tuple of tensors, the input tensor and the outout one
    '''
    # Convert the source text to the input tensor
    input_tensor = tensorFromSentence(input_lang, pair[0])
    # Convert the summary to the output tensor
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    ''' Run all the steps in the training phase of a batch of examples
        Input:
        - input_tensor: a tensor, vector representation of the input text
        - target_tensor: a tensor, vector representation of the expect or labelled output or summary
        - encoder: a Class Encoder object, the encoder
        - decoder: a Class AttnDecoder object, the decoder
        - encoder_optimizer: a torch optimizer, the optimizer of the encoder
        - decoer_optimizer: a torch optimizer, the optimizer of the decoder
        - criterion: a pytoch loss function
        - max_length: an integer, maximun length of an output
    '''
    #Init the encoder hidden state
    encoder_hidden = encoder.initHidden()
    
    # Reset the optimizer
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Set the length if the source text and the summary
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # Create the initial encoder output, all zeros
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    # For every token in the source text or inout
    for ei in range(input_length):
        # Forward pass of the encoder to get the encoder output and hidden state
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        
    # Set the initial decoder input as the SOS token
    decoder_input = torch.tensor([[SOS_token]], device=device)
    #Set the initial decoder hidden state equals to the last encoder hidden state
    decoder_hidden = encoder_hidden

    # Active teacher forcing with probability teacher_forcing_ratio 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # Forward pass of the decoder returning the decoder output, hidden state and context vector
            # of the attention mechanism
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Increment the loss function by the loss of the decoder output in the actual time step
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # Forward pass of the decoder returning the decoder output, hidden state and context vector
            # of the attention mechanism
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
             # Select the decoder output with the highest probability
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # Increment the loss function by the loss of the decoder output in the actual time step
            loss += criterion(decoder_output, target_tensor[di])
            # Stop training if the EOS token is returned
            if decoder_input.item() == EOS_token:
                break
   # Apply the backward pass to calculate and propagate the loss
    loss.backward()
    # Apply a step of the optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    # Return the final loss
    return loss.item() / target_length
import time
import math


def asMinutes(s):
    ''' Return the seconds, s, to a string in the format: Xm Ys'''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    ''' Return '''
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    ''' Plot the points in a line graph to show a training metric'''
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    ''' Train a encoder-decoder model on the input x for n_iters iterations
        Input:
        - encoder: a Class Encoder object, the encoder
        - decoder: a Class AttnDecoder object, the decoder
        - x: array of strings, source texts of the training dataset
        - y: array of strings, target texts or summaries of the training dataset
        - vocab_input: a Vocab Class object, vocabulary of the source texts
        - vocab_output: a Vocab Class object, vocabulary of the target texts
        - n_iters: integer, number of iterations
        - print_every: integer, print the progress every print_every iteration
        - plot_every: integer, plot the losses every plot_every iteration
        - learning_rate: float, learning rate
    '''

    print("Training....")
    # Get the current time
    start = time.time()
    # Initialize variables for progress tracking
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # Create the optimizer for the encoder and the decoder
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # Extract the training set randomly for all the iterations
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    # Set the function loss to apply
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        if iter% 1000 == 0:
            print(iter,"/",n_iters + 1) # Plot progress
            
        # Get the next pair of source text and target to train on
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # Train on the pair of data selected
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        # Set the variable to plot the progress
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            # Print the ETA and current loss
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            # Plot the current loss
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
def predict(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    ''' Function to predict the summary of the source text sentence with a max length
        Input:
        - encoder: a Class Encoder object, the encoder
        - decoder: a Class AttnDecoder object, the decoder
        - vocab_input: a Vocab Class object, vocabulary of the source texts
        - vocab_output: a Vocab Class object, vocabulary of the target texts
        - sentence: string, source text to predict

    '''
    with torch.no_grad():
        # Get the tensor of the source text
        input_tensor = tensorFromSentence(input_lang, sentence)
        # Calculate the length of the source text
        input_length = input_tensor.size()[0]
        # Set the initial hidden state of the encoder
        encoder_hidden = encoder.initHidden()
        # Set the initial encoder outputs
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # For every word in the input
        for ei in range(input_length):
            # Forward pass of the encoder
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        # Set the initial input of the decoder 
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        # Set the initial hidden state of the decoder to the hidden state of the decoder in the last time step
        decoder_hidden = encoder_hidden

        decoded_words = []
        # Set the initial context vectors of the decoder to zeros
        decoder_attentions = torch.zeros(max_length, max_length)
        # For every word or step in the output sequence
        for di in range(max_length):
            # Forward pass of the decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Save the decoder attention vector of the step
            decoder_attentions[di] = decoder_attention.data
            # Get the element in the decoder output with the highest probability (the best output)
            topv, topi = decoder_output.data.topk(1)
            # If the token returned is EOS then finish
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                # Append the token in the summary returned by the decoder
                decoded_words.append(output_lang.index2word[topi.item()])
            # Set the decoder input to the output selected
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
def generate_predictions(x_test, encoder, decoder, input_vocab, output_vocab, max_length, print_every=20):
    ''' Generate the predicted summaries of the source texts on x_test
        Input:
        - x_test: list of strings, the source texts
        - encoder: a Class Encoder object, the encoder
        - decoder: a Class AttnDecoder object, the decoder
        - input_vocab: a Vocab Class object, vocabulary of the source texts
        - output_vocab: a Vocab Class object, vocabulary of the target texts
        - max_length: integer, max length of the output summary
        - print_every: integer, print progress every print_every iterations
    '''
    predicted_summaries = []
    # Set a progress bar
    #kbar = pkbar.Kbar(target=len(x_test), width=8)
    # Para cada text or document in the validation dataset
    for i,doc in enumerate(x_test):
        # Predict the summary for the document
        #pred_summ = predict(doc,vocab,params,batch_size=1)
        pred_summ,_ = predict(encoder, decoder, doc, input_vocab, output_vocab, max_length)
        predicted_summaries.append(' '.join(pred_summ[:-1]))
        #predicted_summaries.append(' '.join(pred_summ))
        
        #if i%print_every==0:
        #    kbar.update(i)
            
    # Set teh labeled summaries as the y_test variable, column summary of our dataset
    return predicted_summaries

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = predict(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
hidden_size = 100
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.2).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
torch.save(encoder1.state_dict(), './enc.w')
torch.save(attn_decoder1.state_dict(), './att.w')
# Save the vocabularies
input_lang.save_to_file('input_vocab.pkl')
output_lang.save_to_file('output_vocab.pkl')
# Load the vocabularies, 
#input_vocab= load_vocab('input_vocab.pkl')
#output_vocab= load_vocab('output_vocab.pkl')
# Uncomment and execute if you want to show some results quickly
#evaluateRandomly(encoder1, attn_decoder1)
x_test = valid_dataset['text'].values
y_test = valid_dataset['summary'].values
# Generate the predctions on the validation dataset
predicted_summaries = generate_predictions(x_test, encoder1, attn_decoder1, input_lang, output_lang, MAX_LENGTH, 100)
# Set teh labeled summaries as the y_test variable, column summary of our dataset
labeled_summaries = y_test
#print(len(x_test), len(labeled_summaries), len(predicted_summaries))
print('\n Pred: ',predicted_summaries[100],'\n Target: ', labeled_summaries[100])
print('\n Pred: ',predicted_summaries[200],'\n Target: ', labeled_summaries[200])
print('\n Pred: ',predicted_summaries[300],'\n Target: ', labeled_summaries[300])
print('\n Pred: ',predicted_summaries[400],'\n Target: ', labeled_summaries[400])
print('\n Pred: ',predicted_summaries[500],'\n Target: ', labeled_summaries[500])
def save_textfile(filename, strings):
    ''' Save the contect of a list of strings to a file called filename
    
        Input:
           - filename: name of the file to save the strings
           - strings: a list of string to save to disk
    '''
    
    with open(filename, 'w') as f:
        for item in strings:
            #Remove any \n in the string
            item = remove_CTL(item)
            f.write("%s\n" % item)

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
# Calculate the Rouge-2 and Rouge-L metrics for the validation dataset
r2_f, r2_p, r2_r, rl_f, rl_p, rl_r = eval_metrics(predicted_summaries, list(labeled_summaries), False)
print('Mean Rouge-2 FScore: ',np.mean(r2_f), 'Mean Rouge-L FScore: ',np.mean(rl_f))
#Store the results on the dataframe
valid_dataset['pred_summary'] = predicted_summaries
valid_dataset['rouge2-f'] = r2_f
valid_dataset['rouge2-p'] = r2_p
valid_dataset['rouge2-r'] = r2_r
valid_dataset['rougel-f'] = rl_f
valid_dataset['rougel-p'] = rl_p
valid_dataset['rougel-r'] = rl_r
valid_dataset.to_csv('results.csv', index=False)
valid_dataset.head(5)
# Plot
kwargs = dict(hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
# plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, dpi=100)
sns.distplot(valid_dataset['rouge2-f'] , color="dodgerblue", ax=axes[0], axlabel='Rouge-2 Fscore')
sns.distplot(valid_dataset['rougel-f'], color="deeppink", ax=axes[1], axlabel='Rouge-L Fscore')
