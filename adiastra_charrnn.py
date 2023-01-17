import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
with open('/kaggle/input/charrnn/data/data/anna.txt', 'r') as f:
    text = f.read()
text[:500]
"""Creating two dictonaries
   1. int2char : which maps integers to characters
   2. char2int : which maps charaters to integers"""

chars = tuple(set(text))
int2char = dict(enumerate((chars)))
char2int = {ch: ii for ii, ch in int2char.items()}

#ENCODING THE TEXT:
encoded = np.array([char2int[ch] for ch in text])
encoded[:100]
def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((arr.size, n_labels), dtype = np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot
test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)

print(one_hot)
def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size*seq_length
    n_batches = len(arr)//batch_size_total
    
    arr = arr[:n_batches*batch_size_total]
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:,:-1], y[:,-1] = x[:,1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y 
batches = get_batches(encoded, 8, 50)
x, y = next(batches)
print('x/n', x[:10, :10])
print('\ny\n', y[:10, :10])
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
        
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden       
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            h = tuple([each.data for each in h])

            net.zero_grad()
            
            output, h = net(inputs, h)
            
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y                                         
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train()
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)
batch_size = 128
seq_length = 100
n_epochs = 30
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
model_name = 'rnn_30_epoch.net'
checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens' : net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)
def predict(net, char, h = None, top_k = None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    
    
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)
    
    p = F.softmax(out, dim =1).data
    
    #GETTING TOP CHARACTERS
    
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    
    return net.int2char[char], h
def sample(net, size, prime='The', top_k=None):
 
    net.eval() 
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
print(sample(net, 1000, prime='Aditya ', top_k=5))
with open('rnn_30_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)
    
loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])
print(sample(loaded, 2000, top_k = 10, prime="This is Geekquad, follow me on https://github.com/geekquad "))
