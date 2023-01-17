import re
import string
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
np.random.seed(0)
torch.manual_seed(0)
with open('../input/mahabharat-english/oliver.txt', 'r') as file:
    mahabharat_text_raw = file.read()
type(mahabharat_text_raw)
mahabharat_text_raw
def clean_data(data):    
    # removing ï»¿SECTION
    data = re.sub('SECTION [I-X]*', '', data)
    
    # Removing \n
    data = re.sub(r'\n', ' ', data)
    
    # removing extra symbols
    data = re.sub("[^\w]", ' ', data)
    data = data.replace(r'--','  ')
    data = re.sub(' +', ' ', data)
    tokens = data.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w.lower() for w in tokens]
 
    
    return data
mahabharat_text_cleaned = clean_data(mahabharat_text_raw)
mahabharat_text_cleaned
mahabharat_text_cleaned_split = mahabharat_text_cleaned.split()
print('Total word in Mahabharat = ',len(mahabharat_text_cleaned_split))
def create_seq(text):
    length = 50
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        line = ' '.join(seq)
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
    return sequences
mahabharat_seq = create_seq(mahabharat_text_cleaned_split)
n_vocab = len(set(mahabharat_text_cleaned_split))
n_vocab
len(mahabharat_seq[0])
# Tokenizing based on vocab and dict
word_counts = Counter(mahabharat_text_cleaned_split)
sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
vocab_to_int = {w: k for k, w in int_to_vocab.items()}
n_vocab = len(int_to_vocab)
n_vocab
# Tokenizing sequenced text
t0 = time.time()
mahabharat_int_text = []
for seq in mahabharat_seq:
    int_text = [vocab_to_int[w] for w in seq.split()]
    mahabharat_int_text.append(int_text)
print('{} seconds'.format(time.time() - t0))
sequences_array = np.array(mahabharat_int_text)
sequences_array.shape
"""
# Tokenization
tokenizer = Tokenizer(num_words=n_vocab, filters='@')
tokenizer.fit_on_texts(mahabharat_seq)
sequences = tokenizer.texts_to_sequences(mahabharat_seq)
sequences_array = np.array(sequences)
"""
X, y = sequences_array[:, :-1], sequences_array[:,-1]
# one-hot encoding y
def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot
y = one_hot_encode(y, n_vocab)
seq_size = X.shape[0]//16
#Limiting data size for 32 batches
X_new = X[:seq_size]
y_new = y[:seq_size]
# List of x and y
X_list = X_new
y_list = y_new
X_tensor = torch.Tensor(X_list)
y_tensor = torch.Tensor(y_list)
X_tensor.shape
seq_size = 50
embedding_size = 64
lstm_size = 128
batch_size = 16
lr = 0.003
drop_prob = 0.1
n_layers = 3
my_dataset = TensorDataset(X_tensor, y_tensor)
train_set = DataLoader(my_dataset, batch_size=batch_size, drop_last=True)
train_check = torch.cuda.is_available()
if train_check:
    print('Training on GPU')
else:
    print('Training on CPU')
class Mahabharat_Model(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size,drop_prob, n_layers):
        super(Mahabharat_Model, self).__init__()
        
        # initializing variables
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        
        # Initializing structure
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, n_layers, dropout=drop_prob, batch_first=True)
        self.dense0 = nn.Linear(lstm_size, n_vocab)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)

        
        logits = self.dense0(output)
        
        return logits, state
    
    def zero_state(self, batch_size):
        return(torch.zeros(3, batch_size, self.lstm_size), torch.zeros(3, batch_size, self.lstm_size))
model = Mahabharat_Model(n_vocab, seq_size, embedding_size, lstm_size, drop_prob, n_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Loss and optimizers
critertion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 150
for e in range(epochs):
    print('Epoch running:', e)
    state_h, state_c = model.zero_state(batch_size)
    
    # transfer data to gpu if available
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    # Claculating time of each epoch
    t0 = time.time()
    
    for x, y in train_set:
        
        # Training mode
        model.train()
        
        # Reset all gradients
        model.zero_grad()
        
        # Transfer data to gpu
        x = x.to(device)
        y = y.to(device)
        
        logits, (state_h, state_c) = model(x.long(), (state_h, state_c))
        
        # loss calculation
        loss = critertion(logits, y.long())
        
        state_h = state_h.detach()
        state_c = state_c.detach()
        
        loss_value = loss.item()
        
        # Backprop
        loss.backward()
        
        # Gradient clipping
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # optimizer step
        optimizer.step()
    
    print('Time For 1 epoch :', time.time() - t0)    
    print('Epoch: {}/{}'.format(e, epochs),'Loss: {}'.format(loss_value))
torch.save(model, 'model.pth')
PATH = 'model.pth'
net = torch.load(PATH, map_location=device)
net.eval()
def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    model.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix.long(), (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
    
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix.long(), (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))
predict(device=device, net=model, words=["faithful", "specimen"], n_vocab=n_vocab, vocab_to_int=vocab_to_int, int_to_vocab=int_to_vocab)

