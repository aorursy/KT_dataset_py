import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
lstm = nn.LSTM(3, 3) #input dimension and output dimension are 3
inputs = [ torch.randn(1, 3) for _ in range(5) ] #make a sequence of length 5

hidden = ( torch.randn(1, 1, 3), torch.randn(1, 1, 3) )
for i in inputs:
    out, hidden = lstm( i.view(1,1, -1), hidden ) #torch requires it's inputs to be 3d tensors
    #out gives access to all hidden states in network from the past, while hidden returnes only the last one

inputs = torch.cat( inputs).view( len(inputs), 1, -1) #changes a list of tensors into a tensor
hidden = ( torch.randn(1, 1, 3), torch.randn(1, 1, 3) ) #clean out the hidden state
out, hidden = lstm(inputs, hidden)
#w1, w2, w3... are our input sentence, where w.i E v - our vocabulary
#T - tag set
#y.i - tag of word w.i
#y'.i - prediction tag for word w.i

#Assign each tag a unique index, like in BOW we had wrd_to_ix. Pass the sentence through LSTM. Call the hidden state at the timestamp i, h.i
#Our prediction is y'.i = argmax_j( log_softmax( Ah.i + b)_j )
def prepare_sequence( seq, to_ix ):
    idxs = [ to_ix[w] for w in seq ]
    return torch.tensor(idxs, dtype = torch.long )
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len( word_to_ix )
print( word_to_ix )
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#these will usually be 32-64 dimensional
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
class LSTMTagger(nn.Module):
    def __init__(self,  embedding_dim, vocab_size, hidden_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim) #lstm takes embeddings of dimension embedding_dim, and outputs hidden layer of size hidden_size
        self.hidden2tag = nn.Linear(hidden_size, tagset_size) #this is how we use a trained hidden state to return tags
        self.hidden = self.init_hidden() 
    
    def init_hidden(self):
        return ( torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim) ) #you create those two hidden states: an immediate one and the long term one
    
    def forward(self, sentence):
        #you look at embeddings of the words in the sentence
        embeds = self.word_embeddings(sentence)#where sentence is a tensor of indexes
        
        lstm_out, self.hidden = self.lstm( embeds.view( len(sentence), 1, -1) , self.hidden) 
        tag_space = self.hidden2tag( lstm_out.view( len(sentence), -1) )
        tag_scores = F.log_softmax( tag_space, dim=1)
        return tag_scores
        
#TRAIN THE MODEL
model = LSTMTagger( EMBEDDING_DIM, len(word_to_ix), HIDDEN_DIM, len(tag_to_ix) )
loss_function = nn.NLLLoss()
optimizer = optim.SGD( model.parameters(), lr = 0.1)

with torch.no_grad():
    inputs = prepare_sequence( training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores) #i, j entry is the score of ith input for the jth tag
    
epochs = 300
for epoch in range(epochs):
    for sentence, tags in training_data:
        #clean gradients
        model.zero_grad()
        #clean history 
        model.hidden = model.init_hidden()
        
        #prepare data
        sentence_in = prepare_sequence( sentence, word_to_ix )
        targets = prepare_sequence( tags, tag_to_ix )
        
        #train
        tag_scores = model(sentence_in)
        
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print( tag_scores)
        
#This work is based on the code and knowledge provided on the website https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html. It contains my interpretation
#of the techniques published on the website.

