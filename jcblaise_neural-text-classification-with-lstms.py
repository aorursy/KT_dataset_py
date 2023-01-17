!pip install sacremoses
import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data_utils

import torch.nn.utils.rnn as rnn_utils

from torch.autograd import Variable



from torchtext.vocab import Vectors

from sklearn.model_selection import train_test_split

from sacremoses import MosesTokenizer

from html import unescape

from tqdm import tqdm



device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

torch.cuda.manual_seed(42);

torch.manual_seed(42);
df = pd.read_csv('../input/imdb-sentiments/train.csv')
df.loc[0]['text']
moses = MosesTokenizer()



def process(s, join=True):

    tokens = [unescape(t) for t in moses.tokenize(s)]

    n_tokens = []

    for i, token in enumerate(tokens):

        if i == 0:

            n_tokens.append('xxbos')

            n_tokens.append(token.lower().replace('.', ''))

        else:

            if token == '.':

                n_tokens.append('xxsep')

            elif token[0] == '\'' or token[0] == '"':

                n_tokens.append('xxquo')

            elif token[0].isupper():

                n_tokens.append('xxmaj')

                n_tokens.append(token.lower().replace('.', ''))

            else:

                n_tokens.append(token.lower().replace('.', ''))

    n_tokens.append('xxeos')

    

    if join:

        return " ".join(n_tokens)

    return n_tokens
# Process all the text

li = []

for text in tqdm(list(df['text'])):

    li.append(process(text))

y = list(df['sentiment'])

    

# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(li, y, random_state=42)
vocab = ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxmaj', 'xxsep', 'xxquo']

word2idx = {vocab[i]:i for i in range(7)}

vocab = set(vocab)

i = 7



# Build the vocabulary lookups

for line in tqdm(X_train):

    tokens = line.split()

    for token in tokens:

        if token not in vocab:

            vocab.add(token)

            word2idx[token] = i

            i += 1

            

# Reverse the lookup table

idx2word = {i:  word for i, word in enumerate(word2idx.keys())}



# Get the vocabulary size and indicate the classes

vocab_sz = len(vocab)

classes = ['positive', 'negative']
def serialize(sentence):

    return torch.LongTensor([word2idx[token] if token in vocab else word2idx['xxunk'] for token in sentence.split()])



# Serialize all data

X_train = [serialize(s) for s in tqdm(X_train)]

X_test = [serialize(s) for s in tqdm(X_test)]
X_train[0]
def make_batches(x, y, padding_value, bs=32, random_state=-1):

    size = len(x)

    batch_size = bs

    if random_state > -1:

        np.random.seed(random_state)

    perm = np.random.permutation(size) # Shuffles list of indices

    

    iterator = []

    

    for i in range(0, size, batch_size):

        batch_idx = perm[i:i+batch_size] # Batches a number of indices equal to bs

        x_ = [x[i] for i in batch_idx]

        y_ = [y[i] for i in batch_idx]

        

        # Sort x based on length in descending order

        x_, y_ = zip(*sorted(zip(x_, y_), key=lambda b: len(b[0]), reverse=True))

        

        # Convert to tensors, and padd the sequences

        l_ = torch.IntTensor([len(b) for b in x_])

        x_ = rnn_utils.pad_sequence(x_, batch_first=True, padding_value=padding_value).t()

        y_ = torch.LongTensor(y_)

    

        iterator.append((x_, y_, l_))

    

    return iterator
batch_size = 64

padding_value = word2idx['xxpad']



train_loader = make_batches(X_train, y_train, padding_value=padding_value, bs=batch_size, random_state=42)

test_loader =  make_batches(X_test, y_test, padding_value=padding_value, bs=batch_size, random_state=42)
x, y, l = train_loader[0]



print(x)

print("x shape:", x.shape)

print(y)

print("y shape:", y.shape)

print(l)

print("l shape:", l.shape)
embedding = nn.Embedding(vocab_sz, 300)
out = embedding(x)

print("Output shape:", out.shape)

print(out)
print("Shape of the first item in the output:", out[0].shape)

print(out[0])
rnn = nn.LSTM(300, 128)
(hidden, cell) = torch.zeros(1, 64, 128), torch.zeros(1, 64, 128)

out = embedding(x)

print("Embedded output shape:", out.shape)



out = rnn_utils.pack_padded_sequence(out, l)

print("Packed output shape:", out.data.shape)



out, (hidden, cell) = rnn(out, (hidden, cell))
hidden.shape
out = hidden[-1, :, :]

print("Final timestep shape:", out.shape)

print(out)
fc = nn.Linear(128, 2) # Initialize with 128 units and 2 output units

out = torch.softmax(fc(out), dim=1)

print("Shape of output:", out.shape)

print(out[:5]) # Print the first five probabilities
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_sz, embedding_dim, hidden_dim, output_dim):

        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_sz, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim

        

    def init_hidden(self, bs):

        return torch.zeros(1, bs, self.hidden_dim), torch.zeros(1, bs, self.hidden_dim)

        

    def forward(self, X, hidden, cell, lengths):

        out = self.embedding(X)

        out = rnn_utils.pack_padded_sequence(out, lengths)

        out, (hidden, cell) = self.rnn(out, (hidden, cell))

        out = hidden[-1, :, :]

        out = torch.log_softmax(self.fc(out), dim=1)

        return out, (hidden, cell)
model = LSTMClassifier(vocab_sz=vocab_sz, embedding_dim=300, hidden_dim=128, output_dim=2)

criterion = nn.CrossEntropyLoss()



# Make a forward pass

(hidden, cell) = model.init_hidden(batch_size)

out, (hidden, cell) = model(x, hidden, cell, l)
def accuracy(y_pred, y_acc):

    with torch.no_grad():

        return torch.sum(torch.max(torch.exp(y_pred), dim=1)[1] == y_acc).item() / len(y_acc)

    

print("Accuracy: {:.4f}".format(accuracy(out, y)))

print("Loss: {:.4f}".format(criterion(out, y)))
model = LSTMClassifier(vocab_sz=vocab_sz, embedding_dim=300, hidden_dim=128, output_dim=2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 4

for e in range(1, epochs + 1):

    train_loss = 0

    train_acc = 0

    

    model.train()

    for batch in tqdm(train_loader, leave=True):

        x, y, lens = batch

        x, y, lens = x.cuda(), y.cuda(), lens.cuda()



        # Initialize the hidden state

        (hidden, cell) = model.init_hidden(x.shape[1])

        hidden = hidden.to(device)

        cell = cell.to(device)

        

        # Forward pass and backprop

        out, (hidden, cell) = model(x, hidden, cell, lens)

        loss = criterion(out, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        train_acc += accuracy(out, y)



    # Scale accuracy and losses

    train_loss /= len(train_loader)

    train_acc /= len(train_loader)



    test_loss = 0

    test_acc = 0

    

    model.eval()

    with torch.no_grad():

        for batch in tqdm(test_loader, leave=True):

            x, y, lens = batch

            x, y, lens = x.cuda(), y.cuda(), lens.cuda()



            (hidden, cell) = model.init_hidden(x.shape[1])

            hidden = hidden.to(device)

            cell = cell.to(device)

            

            out, (hidden, cell) = model(x, hidden, cell, lens)

            loss = criterion(out, y)



            test_loss += loss.item()

            test_acc += accuracy(out, y)



    test_loss /= len(test_loader)

    test_acc /= len(test_loader)

    

    print("Epoch {:4} | Train Loss {:.4f} | Train Acc {:.4f} | Test Loss {:.4f} | Test Acc {:.4f}".format(e, train_loss, train_acc, test_loss, test_acc))
model.cpu()

test = "The movie was good! I liked the characters and the soundtrack. Overall impressive."



# Process the sentence and unsqueeze to make a "batch of 1"

test = torch.LongTensor(serialize(process(test))).unsqueeze(1) 

lengths = torch.LongTensor([len(test)])

(hidden, cell) = model.init_hidden(1)



with torch.no_grad():

    out, _ = model(test, hidden, cell, lengths)

    out = torch.exp(out)

    m = torch.max(out, dim=1)

    print("Prediction: {} | Confidence: {:.4f}".format(classes[m[1].item()], m[0].item()))
x, y, l = train_loader[0]

embedding = nn.Embedding(vocab_sz, 300)

rnn = nn.LSTM(300, 128, bidirectional=True)
(hidden, cell) = torch.zeros(2, 64, 128), torch.zeros(2, 64, 128) # Note the 2 in the first dimension

out = embedding(x)

print("Embedded output shape:", out.shape)



out = rnn_utils.pack_padded_sequence(out, l)

print("Packed output shape:", out.data.shape)



out, (hidden, cell) = rnn(out, (hidden, cell))
hidden.shape
h_cat = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=1)

print("Concatenated shape:", h_cat.shape)
fc = nn.Linear(128 * 2, 2) # Note the * 2

out = torch.softmax(fc(h_cat), dim=1)

print("Final shape:", out.shape)

print(out[:5])
num_layers = 2



x, y, l = train_loader[0]

embedding = nn.Embedding(vocab_sz, 300)

rnn = nn.LSTM(300, 128, num_layers=num_layers)
(hidden, cell) = torch.zeros(num_layers, 64, 128), torch.zeros(num_layers, 64, 128) # We now generalize number of layers

out = embedding(x)

print("Embedded output shape:", out.shape)



out = rnn_utils.pack_padded_sequence(out, l)

print("Packed output shape:", out.data.shape)



out, (hidden, cell) = rnn(out, (hidden, cell))
out = hidden[-1, :, :]



fc = nn.Linear(128, 2) 

out = torch.softmax(fc(out), dim=1)

print("Final shape:", out.shape)

print(out[:5]) # First five probabilities
num_layers = 2

bidirectional = True

recur_drop = 0.3



x, y, l = train_loader[0]

embedding = nn.Embedding(vocab_sz, 300)

rnn = nn.LSTM(300, 128, num_layers=num_layers, bidirectional=bidirectional, dropout=recur_drop)
if bidirectional: # Double the first dimension if bidirectional

    (hidden, cell) = torch.zeros(num_layers * 2, 64, 128), torch.zeros(num_layers * 2, 64, 128)

else:

    (hidden, cell) = torch.zeros(num_layers, 64, 128), torch.zeros(num_layers, 64, 128)
out = embedding(x)

print("Embedded output shape:", out.shape)



out = rnn_utils.pack_padded_sequence(out, l)

print("Packed output shape:", out.data.shape)



out, (hidden, cell) = rnn(out, (hidden, cell))
if bidirectional:

    out = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=1)

else:

    out = hidden[-1, :, :]



fc = nn.Linear(128 * 2, 2) if bidirectional else nn.Linear(128, 2) # Branch if bidirectional

out = torch.softmax(fc(out), dim=1)

print("Final shape:", out.shape)

print(out[:5]) # First five probabilities
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_sz, embedding_dim, hidden_dim, output_dim, bidirectional, rnn_layers, recur_dropout=0.3):

        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_sz, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=rnn_layers, dropout=recur_dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim) if bidirectional else nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim

        

    def init_hidden(self, bs):

        if self.rnn.bidirectional:

            return torch.zeros(self.rnn.num_layers * 2, bs, self.hidden_dim), torch.zeros(self.rnn.num_layers * 2, bs, self.hidden_dim)

        else:

            return torch.zeros(self.rnn.num_layers, bs, self.hidden_dim), torch.zeros(self.rnn.num_layers, bs, self.hidden_dim)

        

    def forward(self, X, hidden, cell, lengths):

        out = self.embedding(X)

        out = rnn_utils.pack_padded_sequence(out, lengths)

        out, (hidden, cell) = self.rnn(out, (hidden, cell))

        h_cat = torch.cat([ hidden[-2, :, :], hidden[-1, :, :] ], dim=1) if self.rnn.bidirectional else hidden[-1, :, :]

        out = torch.log_softmax(self.fc(h_cat), dim=1)

        return out, (hidden, cell)
model = LSTMClassifier(vocab_sz, embedding_dim=300, hidden_dim=128, output_dim=2, 

                       bidirectional=True, rnn_layers=2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 4

for e in range(1, epochs + 1):

    train_loss = 0

    train_acc = 0

    for batch in tqdm(train_loader, leave=True):

        x, y, lens = batch

        x = x.cuda()

        y = y.cuda()

        lens = lens.cuda()



        (hidden, cell) = model.init_hidden(x.shape[1])

        hidden = hidden.to(device)

        cell = cell.to(device)

        out, (hidden, cell) = model(x, hidden, cell, lens)

        loss = criterion(out, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        train_acc += accuracy(out, y)



    train_loss /= len(train_loader)

    train_acc /= len(train_loader)



    test_loss = 0

    test_acc = 0

    with torch.no_grad():

        for batch in tqdm(test_loader, leave=True):

            x, y, lens = batch

            x = x.cuda()

            y = y.cuda()

            lens = lens.cuda()



            (hidden, cell) = model.init_hidden(x.shape[1])

            hidden = hidden.to(device)

            cell = cell.to(device)

            out, (hidden, cell) = model(x, hidden, cell, lens)

            loss = criterion(out, y)



            test_loss += loss.item()

            test_acc += accuracy(out, y)



    test_loss /= len(test_loader)

    test_acc /= len(test_loader)

    

    print("Epoch {:4} | Train Loss {:.4f} | Train Acc {:.4f} | Test Loss {:.4f} | Test Acc {:.4f}".format(e, train_loss, train_acc, test_loss, test_acc))
model.cpu()



test = "That movie was awful."

test = torch.LongTensor(serialize(process(test))).unsqueeze(1)

lengths = torch.LongTensor([len(test)])

(hidden, cell) = model.init_hidden(1)



with torch.no_grad():

    out, _ = model(test, hidden, cell, lengths)

    out = torch.exp(out)

    m = torch.max(out, dim=1)

    print("Prediction: {} | Confidence: {:.4f}".format(classes[m[1].item()], m[0].item()))
v = Vectors('../input/glove840b300dtxt/glove.840B.300d.txt')
out = v['hello']

print("Embedding of 'hello' dimensions:", out.shape)

print(out)
m = (v['man'] - v['king']).unsqueeze(0)

w = (v['woman'] - v['queen']).unsqueeze(0)



l2_dist = nn.PairwiseDistance(p=2)

out = l2_dist(m, w)



print("L2 Distance:", out.item())
emb = nn.Embedding(vocab_sz, 300)

batch = []

for i in range(vocab_sz):

    batch.append(v[idx2word[i]])

weights = torch.stack(batch)

emb.weight.data.copy_(weights)

emb.weight.requires_grad = False
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_sz, embedding_dim, hidden_dim, output_dim, bidirectional, rnn_layers, recur_dropout=0.3, pretrained=False, pretrained_emb=None):

        super(LSTMClassifier, self).__init__()

        if pretrained and pretrained_emb is not None:

            self.embedding = pretrained_emb

        else:

            self.embedding = nn.Embedding(vocab_sz, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=rnn_layers, dropout=recur_dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim) if bidirectional else nn.Linear(hidden_dim, output_dim)

        self.hidden_dim = hidden_dim

        

    def init_hidden(self, bs):

        if self.rnn.bidirectional:

            return torch.zeros(self.rnn.num_layers * 2, bs, self.hidden_dim), torch.zeros(self.rnn.num_layers * 2, bs, self.hidden_dim)

        else:

            return torch.zeros(self.rnn.num_layers, bs, self.hidden_dim), torch.zeros(self.rnn.num_layers, bs, self.hidden_dim)

        

    def forward(self, X, hidden, cell, lengths):

        out = self.embedding(X)

        out = rnn_utils.pack_padded_sequence(out, lengths)

        out, (hidden, cell) = self.rnn(out, (hidden, cell))

        h_cat = torch.cat([ hidden[-2, :, :], hidden[-1, :, :] ], dim=1) if self.rnn.bidirectional else hidden[-1, :, :]

        out = torch.log_softmax(self.fc(h_cat), dim=1)

        return out, (hidden, cell)
model = LSTMClassifier(vocab_sz, embedding_dim=300, hidden_dim=128, output_dim=2, 

                       bidirectional=True, rnn_layers=2, pretrained=True, pretrained_emb=emb).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 4

for e in range(1, epochs + 1):

    train_loss = 0

    train_acc = 0

    for batch in tqdm(train_loader, leave=True):

        x, y, lens = batch

        x = x.cuda()

        y = y.cuda()

        lens = lens.cuda()



        (hidden, cell) = model.init_hidden(x.shape[1])

        hidden = hidden.to(device)

        cell = cell.to(device)

        out, (hidden, cell) = model(x, hidden, cell, lens)

        loss = criterion(out, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        train_acc += accuracy(out, y)



    train_loss /= len(train_loader)

    train_acc /= len(train_loader)



    test_loss = 0

    test_acc = 0

    with torch.no_grad():

        for batch in tqdm(test_loader, leave=True):

            x, y, lens = batch

            x = x.cuda()

            y = y.cuda()

            lens = lens.cuda()



            (hidden, cell) = model.init_hidden(x.shape[1])

            hidden = hidden.to(device)

            cell = cell.to(device)

            out, (hidden, cell) = model(x, hidden, cell, lens)

            loss = criterion(out, y)



            test_loss += loss.item()

            test_acc += accuracy(out, y)



    test_loss /= len(test_loader)

    test_acc /= len(test_loader)

    

    print("Epochs {:4} | Train Loss {:.4f} | Train Acc {:.4f} | Test Loss {:.4f} | Test Acc {:.4f}".format(e, train_loss, train_acc, test_loss, test_acc))
model.cpu()



test = "I loved the movie! The cinematography was terrific and the actors were great."

test = torch.LongTensor(serialize(process(test))).unsqueeze(1)

lengths = torch.LongTensor([len(test)])

(hidden, cell) = model.init_hidden(1)



with torch.no_grad():

    out, _ = model(test, hidden, cell, lengths)

    out = torch.exp(out)

    m = torch.max(out, dim=1)

    print("Prediction: {} | Confidence: {:.4f}".format(classes[m[1].item()], m[0].item()))