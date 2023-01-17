### Options

from argparse import Namespace

args = Namespace()

args.window_size = 5 # skip-gram window

args.pad = 32000 # pad idx
args.bos = 32001 # BoS idx
args.eos = 32002 # EoS idx

args.arch = 'GRU' # One of 'LSTM', 'BiLSTM', 'GRU', 'BiGRU'

args.vocab_size = 32768
args.batch_size = 2500
args.learning_rate = 0.05
args.momentum = 0.9
args.epochs = 100

args.embed_size = 512
args.hidden_size = args.embed_size
args.num_layers = 2
args.dropout = 0.2

args.threshold = 0.5 # Classify as dissimilar if under threshold
args.upper_limit = 1 # Eliminate loss if over limit (not currently used)

args.skip_validation = False # Set True to skip validation
import random


class Dataset():
    def __init__(self, labels=True):
        self.idx = []
        self.sent1 = []
        self.sent2 = []
        self.label = [] if labels else None
        self.vocab = None
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            if self.label:
                return zip(self.idx, self.sent1, self.sent2, self.label)[key]
            else:
                return zip(self.idx, self.sent1, self.sent2)[key]
        elif isinstance(key, str):
            if key in ['id', 'idx']:
                return self.idx
            elif key in ['sent1', 'sentence1']:
                return self.sent1
            elif key in ['sent2', 'sentence2']:
                return self.sent2
            elif key == 'label' and self.label != None:
                return self.label
            else:
                raise KeyError
        else:
            raise TypeError
    
    def __iter__(self):
        if self.label != None:
            return zip(self.idx, self.sent1, self.sent2, self.label).__iter__()
        else:
            return zip(self.idx, self.sent1, self.sent2).__iter__()
    
    def construct_vocab(self):
        self.vocab = set()
        for sent in self.sent1:
            for token in sent:
                self.vocab.add(token)
        
        for sent in self.sent2:
            for token in sent:
                self.vocab.add(token)
    
    def append(self, idx, sent1, sent2, label=None):
        if self.label != None and label == None:
            raise TypeError('label not passed to labeled dataset')
        if self.label == None and label != None:
            raise TypeError('label passed to unlabeled dataset')
        
        self.idx.append(idx)
        self.sent1.append(sent1)
        self.sent2.append(sent2)
        if label != None:
            self.label.append(label)
    
    def shuffle(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        
        self.idx = [self.idx[indices[i]] for i in range(len(self))]
        self.sent1 = [self.sent1[indices[i]] for i in range(len(self))]
        self.sent2 = [self.sent2[indices[i]] for i in range(len(self))]
        if self.label != None:
            self.label = [self.label[indices[i]] for i in range(len(self))]
    
    def split(self, n=None, frac=None, rand_sample=True):
        """Splits self into two dataset objects. The arguments specify the size of the second dataset."""
        
        assert(n and not frac or not n and frac)
        
        if frac:
            n = int(len(self) * frac)
        
        indices = list(range(len(self)))
        if rand_sample:
            random.shuffle(indices)
        
        right = sorted(indices[len(self) - n:])
        
        if self.label != None:
            dataset = Dataset(labels=True)
            for i in right:
                dataset.append(self.idx[i], self.sent1[i], self.sent2[i], self.label[i])
            for i in reversed(right):
                del self.idx[i]
                del self.sent1[i]
                del self.sent2[i]
                del self.label[i]
        else:
            dataset = Dataset(labels=False)
            for i in right:
                dataset.append(self.idx[i], self.sent1[i], self.sent2[i])
            for i in reversed(right):
                del self.idx[i]
                del self.sent1[i]
                del self.sent2[i]
        
        return self, dataset
                

def read_data(filename):
    infile = open(filename, 'r')
    l = len(infile.readline().strip().split(','))
    
    dataset = Dataset(labels=(l >= 4))
    
    for line in infile:
        vals = line.strip().split(',')
        if len(vals) >= 3:
            idx = int(vals[0])
            sent1 = [int(token) for token in vals[1].split()]
            sent2 = [int(token) for token in vals[2].split()]
            label = None
        if len(vals) >= 4:
            label = int(vals[3])
        
        dataset.append(idx, sent1, sent2, label)
    
    dataset.construct_vocab()
    
    return dataset

train = read_data('/kaggle/input/pn2yx4fvd6m3ci9/train.csv')
test = read_data('/kaggle/input/pn2yx4fvd6m3ci9/test.csv')
### Dataset statistics

# Construct an equivalence set
sent_to_sem = {}
sem_to_sent = []

for sent1, sent2, label in zip(train.sent1, train.sent2, train.label):
    sent1 = ' '.join(str(token) for token in sent1)
    sent2 = ' '.join(str(token) for token in sent2)
    if label == 1:
        if sent1 in sent_to_sem:
            i = sent_to_sem[sent1]
            sem_to_sent[i].add(sent2)
            sent_to_sem[sent2] = i
        elif sent2 in sent_to_sem:
            i = sent_to_sem[sent2]
            sem_to_sent[i].add(sent1)
            sent_to_sem[sent1] = i
        else:
            sent_to_sem[sent1] = sent_to_sem[sent2] = len(sem_to_sent)
            sem_to_sent.append(set([sent1, sent2]))
    else:
        if sent1 not in sent_to_sem:
            sent_to_sem[sent1] = len(sem_to_sent)
            sem_to_sent.append(set([sent1]))
        if sent2 not in sent_to_sem:
            sent_to_sem[sent2] = len(sem_to_sent)
            sem_to_sent.append(set([sent2]))

print(f'Unique sentences: {len(sent_to_sem)}') # Number of unique sentences
print(f'Unique meanings: {len(sem_to_sent)}') # Number of unique meanings
print(f'Max length: {max(max(len(sent) for sent in train.sent1), max(len(sent) for sent in train.sent2))}')
print(f'Min length: {min(min(len(sent) for sent in train.sent2), min(len(sent) for sent in train.sent2))}')
print(f'Mean length: {(sum(len(sent) for sent in train.sent1) + sum(len(sent) for sent in train.sent2)) / (2 * len(train)):.2f}')
print()

print(f'Train vocab: {len(train.vocab)}, max idx: {max(train.vocab)}')
print(f'Test vocab: {len(test.vocab)}, max idx: {max(test.vocab)}')

unk_count = 0
for token in test.vocab:
    if token not in train.vocab:
        unk_count += 1
print(f'Unknown tokens in test set: {unk_count}')
print()

if not args.skip_validation:
    train, valid = train.split(n=2500)

print('Train:', len(train))
if not args.skip_validation:
    print('Valid:', len(valid))
print('Test:', len(test))
### Build model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        
        self.arch = args.arch
        
        self.batch_size = args.batch_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = 0 if args.num_layers == 1 else args.dropout
        
        self.embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=args.pad)
        
        if args.arch == 'LSTM' or self.arch == 'BiLSTM':
            self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, 
                                dropout=self.dropout, num_layers=self.num_layers, batch_first=True,
                                bidirectional=(args.arch == 'BiLSTM'))
        elif args.arch == 'GRU' or self.arch == 'BiGRU':
            self.gru = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size, 
                                dropout=self.dropout, num_layers=self.num_layers, batch_first=True,
                                bidirectional=(args.arch == 'BiGRU'))
    
    def forward(self, sent, length):
        emb = self.embed(sent)
        emb = nn.utils.rnn.pack_padded_sequence(emb, length, batch_first=True, enforce_sorted=False)
        
        if self.arch == 'LSTM' or self.arch == 'BiLSTM':
            output, (hidden, cell) = self.lstm(emb, None)
            if self.arch == 'BiLSTM':
                hidden = hidden.view(self.num_layers, 2, self.batch_size, self.hidden_size)
                hidden = torch.cat((hidden[:, 0], hidden[:, 1]), 2)
        elif self.arch == 'GRU' or self.arch == 'BiGRU':
            output, hidden = self.gru(emb, None)
            if self.arch == 'BiGRU':
                hidden = hidden.view(self.num_layers, 2, self.batch_size, self.hidden_size)
                hidden = torch.cat((hidden[:, 0], hidden[:, 1]), 2)
        
        #output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        
        return hidden


class Siamese(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = args.device
        
        self.pad = args.pad
        
        self.encoder = RNN(args)
        
        self.linear_classifier = nn.Linear(args.hidden_size * (4 if args.arch.startswith('Bi') else 2), 1)
    
    def exp_neg_L1(self, rep1, rep2):
        return torch.exp(-torch.sum(torch.abs(rep1 - rep2), 1))
    
    def forward(self, sent1, sent2):
        assert sent1.size() == sent2.size()
        
        len1 = torch.squeeze(torch.LongTensor([s[s != self.pad].size() for s in sent1]).to(self.device))
        len2 = torch.squeeze(torch.LongTensor([s[s != self.pad].size() for s in sent2]).to(self.device))
        
        out1 = self.encoder(sent1, len1)
        out2 = self.encoder(sent2, len2)
        
        rep1 = out1[-1] # Last layer
        rep2 = out2[-1] # Last layer
        
        return self.exp_neg_L1(rep1, rep2)
        
        #return self.linear_classifier(torch.cat((rep1, rep2), 1))
### Train model

args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def pad_sent(sent, max_seq_len=20, args=args):
    pad = args.pad
    bos = args.bos
    eos = args.eos
    
    #sent = [bos] + sent + [eos]
    sent += [pad] * (max_seq_len - len(sent))
    
    return sent


def init_weights(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


model = Siamese(args).to(args.device)
model.apply(init_weights)
model.train()

optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)

criterion = nn.MSELoss()
#criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()


best_loss = 10000000 # very big number
best_params = None
for epoch in range(args.epochs):
    train.shuffle()
    
    sum_loss = 0  # Loss for each epoch
    correct_count = 0 # No. of correct classifications
    tp = fp = tn = fn = 0
    model.train()
    for i in range(0, len(train), args.batch_size):
        sent1 = train.sent1[i:i + args.batch_size]
        sent2 = train.sent2[i:i + args.batch_size]
        label = train.label[i:i + args.batch_size]
        
        optimizer.zero_grad()
        sent1 = torch.LongTensor([pad_sent(sent) for sent in sent1]).to(args.device)
        sent2 = torch.LongTensor([pad_sent(sent) for sent in sent2]).to(args.device)
        
        output = model(sent1, sent2)
        output = torch.squeeze(output)
        
        label = torch.LongTensor(label).to(args.device)
        true_positive = torch.sum(torch.logical_and((args.threshold <= output), label)).item()
        false_positive = torch.sum(torch.logical_and((args.threshold <= output), torch.logical_not(label))).item()
        false_negative = torch.sum(torch.logical_and((output < args.threshold), label)).item()
        true_negative = torch.sum(torch.logical_and((output < args.threshold), torch.logical_not(label))).item()
        
        try:
            assert true_positive + false_positive + false_negative + true_negative == args.batch_size
        except AssertionError:
            print(f'{true_positive}, {false_positive}, {false_negative}, {true_negative}')
        
        tp += true_positive
        fp += false_positive
        fn += false_negative
        tn += true_negative
        
        current_correct = true_positive + true_negative
        correct_count += current_correct
        
        #label_cut = torch.zeros_like(label).float()
        #label_cut[label == 1] = torch.max(output[label == 1], label[label == 1].float() * args.upper_limit)
        loss = criterion(output, label.float())
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        sum_loss += current_loss
    
    print(f'Epoch {epoch + 1}, Loss: {sum_loss / len(train) * args.batch_size:.5f}, Acc: {correct_count / len(train):.3f} ({correct_count} / {len(train)})')
    
    if not args.skip_validation:
        sum_loss = 0  # Loss for validation
        correct_count = 0 # No. of correct classifications
        model.eval()
        for i in range(0, len(valid), args.batch_size):
            sent1 = valid.sent1[i:i + args.batch_size]
            sent2 = valid.sent2[i:i + args.batch_size]
            label = valid.label[i:i + args.batch_size]

            optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
            sent1 = torch.LongTensor([pad_sent(sent) for sent in sent1]).to(args.device)
            sent2 = torch.LongTensor([pad_sent(sent) for sent in sent2]).to(args.device)

            output = model(sent1, sent2)
            output = torch.squeeze(output)

            label = torch.LongTensor(label).to(args.device)
            true_positive = torch.sum(torch.logical_and((args.threshold <= output), label)).item()
            false_positive = torch.sum(torch.logical_and((args.threshold <= output), torch.logical_not(label))).item()
            false_negative = torch.sum(torch.logical_and((output < args.threshold), label)).item()
            true_negative = torch.sum(torch.logical_and((output < args.threshold), torch.logical_not(label))).item()

            try:
                assert true_positive + false_positive + false_negative + true_negative == args.batch_size
            except AssertionError:
                print(f'{true_positive}, {false_positive}, {false_negative}, {true_negative}')

            tp += true_positive
            fp += false_positive
            fn += false_negative
            tn += true_negative

            current_correct = true_positive + true_negative
            correct_count += current_correct

            #label_cut = torch.zeros_like(label).float()
            #label_cut[label == 1] = torch.max(output[label == 1], label[label == 1].float() * args.upper_limit)
            loss = criterion(output, label.float())

            sum_loss += loss.item()
    
        print(f'Valid loss: {sum_loss / len(valid) * args.batch_size:.5f}, Acc: {correct_count / len(valid):.3f} ({correct_count} / {len(valid)})')
    
    if sum_loss < best_loss:
        best_params = (epoch + 1, model.state_dict())
        best_stats = (tp, fp, fn, tn)
    
    scheduler.step(sum_loss / len(valid) * args.batch_size)

if best_params:
    model.load_state_dict(best_params[1])
    print(f'Loading parameters from epoch {epoch + 1}')
    print(f'(tp, fp, fn, tn) = {best_stats}')
else:
    print(f'Using final model state')
    print(f'(tp, fp, fn, tn) = {(tp, fp, fn, tn)}')
### Predict and output

outfile = open('/kaggle/working/submission.csv', 'w')
outfile.write('id,label\n')

model.eval()
positive_count = 0
for i in range(0, len(test), args.batch_size):
    idx = test.idx[i:i + args.batch_size]
    sent1 = test.sent1[i:i + args.batch_size]
    sent2 = test.sent2[i:i + args.batch_size]

    optimizer.zero_grad()
    sent1 = torch.LongTensor([pad_sent(sent) for sent in sent1]).to(args.device)
    sent2 = torch.LongTensor([pad_sent(sent) for sent in sent2]).to(args.device)

    output = model(sent1, sent2)

    label = torch.squeeze(args.threshold <= output).long().tolist()
    
    positive_count += sum(label)
    for j in range(args.batch_size):
        outfile.write(f'{idx[j]},{label[j]}\n')

outfile.close()

print(f'{positive_count}/{len(test)} positive predictions')