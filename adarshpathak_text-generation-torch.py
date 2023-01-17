DATA_PATH = '/kaggle/input/shakespeare-text/text.txt'
import numpy as np

import torch.nn as nn

import torch

import torchvision

from torchvision import transforms

from torch.nn import functional as F

from tqdm import tqdm_notebook

from collections import Counter 
fp = open(DATA_PATH,'r')

txt = fp.read()
txt = txt.replace('$','')

txt = txt.replace('&','')
print(f'total number of characters in corpus: {len(txt)}')

print(f'total number of unique characters in corpus: {len(set(txt))}')
unique_txt = set(txt)
character_count = Counter(txt)

#character_count
char2idx = {char: key for key,char in enumerate(sorted(unique_txt))}



idx2char = [c for c in sorted(unique_txt)]
batch_size = 64

seq_size = 100

coded_text = [char2idx[c] for c in txt]

n_vocab = len(unique_txt)
def batch_generate(text,batch_size= batch_size,seq_size = seq_size):

    #print(f'vocab_size: {len(char2idx)}')    

    total_batches = int(len(coded_text)/(batch_size*seq_size))

    input_txt = text[:total_batches*batch_size*seq_size]

    output_txt = np.zeros_like(input_txt)

    output_txt[:-1] = input_txt[1:]

    output_txt[-1] = input_txt[0]

    input_txt = np.reshape(input_txt,(batch_size,-1))

    output_txt = np.reshape(output_txt,(batch_size,-1))

    return input_txt,output_txt
def get_batches(data,target,batch_size,seq_size):

    num_batches = np.prod(data.shape) // (seq_size * batch_size)

    for i in range(0, num_batches * seq_size, seq_size):

        yield data[:, i:i+seq_size], target[:, i:i+seq_size]
class RNNModule(nn.Module):

    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):

        super(RNNModule, self).__init__()

        self.seq_size = seq_size

        self.lstm_size = lstm_size

        self.embedding = nn.Embedding(n_vocab, embedding_size)

        self.lstm = nn.LSTM(embedding_size,

                            lstm_size,

                            batch_first=True)

        self.dense = nn.Linear(lstm_size, n_vocab)

        

    def forward(self, x, prev_state):

        embed = self.embedding(x)

        output, state = self.lstm(embed, prev_state)

        logits = self.dense(output)



        return logits, state

    

    def zero_state(self, batch_size):

        return (torch.zeros(1, batch_size, self.lstm_size),

                torch.zeros(1, batch_size, self.lstm_size))
def get_loss_and_train_op(net, lr=0.001):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = RNNModule(n_vocab,seq_size,embedding_size=14,lstm_size=128).to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0

    data,target = batch_generate(coded_text)

    for e in tqdm_notebook(range(10)):

        batches = get_batches(data,target,batch_size,seq_size)

        state_h,state_c = net.zero_state(batch_size)

        state_h = state_h.to(device)

        state_c = state_c.to(device)

        for x,y in batches:

            iteration+=1

            net.train()

            optimizer.zero_grad()

            x = torch.tensor(x).to(device)

            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))

            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()

            state_c = state_c.detach()

            loss_value = loss.item()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(

                net.parameters(), 5)

            optimizer.step()

            

            if iteration % 100 == 0:

                print('Epoch: {}/{}'.format(e, 200),

                      'Iteration: {}'.format(iteration),

                      'Loss: {}'.format(loss_value))

                

            if iteration%2000 ==0:

                predict(device,net,['I',' ','a','m',' '],n_vocab,char2idx,idx2char)

                torch.save(net.state_dict(),

                           'model-{}.pth'.format(iteration))
def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):

    net.eval()

    state_h, state_c = net.zero_state(1)

    state_h = state_h.to(device)

    state_c = state_c.to(device)

    for w in words:

        ix = torch.tensor([[vocab_to_int[w]]]).to(device)

        output, (state_h, state_c) = net(ix, (state_h, state_c))

    

    _, top_ix = torch.topk(output[0], k=top_k)

    choices = top_ix.tolist()

    choice = np.random.choice(choices[0])



    words.append(int_to_vocab[choice])

    

    for _ in range(100):

        ix = torch.tensor([[choice]]).to(device)

        output, (state_h, state_c) = net(ix, (state_h, state_c))



        _, top_ix = torch.topk(output[0], k=top_k)

        choices = top_ix.tolist()

        choice = np.random.choice(choices[0])

        words.append(int_to_vocab[choice])



    print(''.join(words))
main()