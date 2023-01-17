import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

alph = "abcdefghijklmnoprstuwvxyzABCDEFGHIJKLMNOPRSTUWVXYZ"
word_to_ix = {}
tag_to_ix = {}
char_to_ix = {}
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
for c in alph:
    if c not in char_to_ix:
        char_to_ix[c] = len(char_to_ix)
def prepare_sequence(sent, word_to_ix):
    return torch.tensor( [word_to_ix[word] for word in sent], dtype= torch.long)

def prepare_char(word, char_to_ix):
    return torch.tensor( [char_to_ix[ch] for ch in word], dtype = torch.long)

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagspace_size ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        #word part
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_w = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_w2tag = nn.Linear(hidden_dim, tagspace_size)
        self.hidden_w = self.init_hidden()
        
        #character part
        self.char_embedding = nn.Embedding( len(alph), embedding_dim)
        self.lstm_ch = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_ch2tag = nn.Linear(hidden_dim, tagspace_size)
        self.hidden_ch = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim) )
    
    def forward(self,sent_in, words_in):
        #word part
        embeds_w = self.word_embedding(sent_in)
        out_w, self.hidden_w = self.lstm_w(embeds_w.view( len(sent_in), 1, -1), self.hidden_w)
        word_scores = self.hidden_w2tag(out_w.view( len(sent_in), -1) ) #tensor of size len(sent_in)x1xtagspace_size
        
        #char part
        char_scores = 0*word_scores #prepare a tensor of the same dimenionality as words for characters
        for e, word_in in enumerate(words_in):
            self.hidden_ch = self.init_hidden()
            embeds_ch = self.char_embedding(word_in)
            out_ch, self.hidden_ch = self.lstm_ch( embeds_ch.view( len(word_in), 1, -1), self.hidden_ch) #each row of out_ch is meant to "say" how likely is this character to belong to a word with  certain tag
            out_ch_sum = torch.sum(out_ch, dim=0) #now, after summation over the columns, out_ch_sum, says how this word is likely to have a certain tag
            char_scores[e] = self.hidden_ch2tag(out_ch_sum)
        log_probs = F.log_softmax( char_scores + word_scores, dim=1 )
        return log_probs
model = LSTMTagger( len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_ix) )
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.NLLLoss()
epochs = 100
losses = []
for epoch in range(epochs):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden_w = model.init_hidden()
        model.hidden_ch = model.init_hidden()
        
        sent_in = prepare_sequence( sentence, word_to_ix)
        words_in = []
        for word in sentence:
            words_in.append( prepare_char(word, char_to_ix) )
        
        predict = model(sent_in, words_in)
        target = prepare_sequence( tags, tag_to_ix )
        predictions = []
        for vec in predict:
            for e, v in enumerate(vec):
                if v==max(vec):
                    predictions.append(e)
        print("My predictions", predictions)
        print(target)
        
        loss = loss_function(predict, target)
        print("loss", loss)
        losses.append( loss.item() )
        loss.backward()
        optimizer.step()   
        
        print("\n")
#This work is based on the code and knowledge from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html, and contains my interpretation of them.
#This is my solution of the exercise proposed by the owners of the website.
