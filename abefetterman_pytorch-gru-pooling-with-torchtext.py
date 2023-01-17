from torchtext import data
id_label = 'id'
text_label = 'comment_text'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

path='../input/'
train_file = 'train.csv'
test_file = 'test.csv'

embedding_file = '../input/glove6b300dtxt/glove.6B.300d.txt'

# some iterators produce StopIteration, which is no longer a warning, we don't need to hear about it
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import io,os,csv

class ToxicDataset(data.Dataset):
    """Defines a Dataset of columns stored in CSV format."""

    def __init__(self, path, fields, skip_header=True, **kwargs):
        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            reader = csv.reader(f)
                
            if skip_header:
                next(reader)

            examples = [data.Example.fromlist(line, fields) for line in reader]

        super(ToxicDataset, self).__init__(examples, fields, **kwargs)
# Define all the types of fields
# pip install spacy for the tokenizer to work (or remove to use default)
TEXT = data.Field(lower=True, include_lengths=True, fix_length=150, tokenize='spacy')
LABEL = data.Field(sequential=False, use_vocab=False)

# we use the index field to re-sort test data after processing
INDEX = data.Field(sequential=False)

train_fields=[
    (id_label, INDEX),
    (text_label, TEXT)
]
for label in label_cols:
    train_fields.append((label,LABEL))

train_data = ToxicDataset(
            path=f'{path}{train_file}',
            fields=train_fields
        )

test_fields=[
    (id_label, INDEX),
    (text_label, TEXT)
]
test_data = ToxicDataset(
            path=f'{path}{test_file}',
            fields=test_fields
        )
from torchtext.vocab import Vectors
# This will download the glove vectors, see torchtext source for other options
max_size = 30000
TEXT.build_vocab(train_data, test_data, vectors=Vectors(embedding_file), max_size=max_size)
INDEX.build_vocab(test_data)

# print vocab information
ntokens = len(TEXT.vocab)
print('ntokens', ntokens)
train = data.BucketIterator(train_data, batch_size=32,
                                sort_key=lambda x: len(x.comment_text),
                                sort_within_batch=True, repeat=False)
test = data.BucketIterator(test_data, batch_size=128,
                                sort_key=lambda x: len(x.comment_text),
                                sort_within_batch=True, train=False, repeat=False)

def get_text(batch):
    return getattr(batch, text_label)
def get_labels(batch):
    # Get the labels as one tensor from the batch object
    return torch.cat([getattr(batch, label).unsqueeze(1) for label in label_cols], dim=1).float()

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nout, nlayers, dropemb=0.2, droprnn=0.0, bidirectional=True):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.drop = nn.Dropout2d(dropemb)
        self.ndir = 2 if bidirectional else 1
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid*self.ndir, nhid, 1, dropout=droprnn, bidirectional=bidirectional) for l in range(nlayers)]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid*self.ndir, nhid, 1, dropout=droprnn, bidirectional=bidirectional) for l in range(nlayers)]
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.decoder = nn.Linear(nhid*self.ndir*2, nout)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, lengths=None):
        emb = self.encoder(input)
        
        raw_output = self.drop(emb)
        
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, lengths)
            
        for rnn in self.rnns:
            raw_output,_ = rnn(raw_output)
        
        if lengths is not None:
            raw_output, lengths = nn.utils.rnn.pad_packed_sequence(raw_output)
            
        bsz = raw_output.size(1)
        rnn_avg = self.avg_pool(raw_output.permute(1,2,0))
        rnn_max = self.max_pool(raw_output.permute(1,2,0))
        rnn_out = torch.cat([rnn_avg.view(bsz,-1),rnn_max.view(bsz,-1)], dim=1)
            
        result = self.decoder(rnn_out)
        return self.decoder(rnn_out)
use_cuda = torch.cuda.is_available()
nhidden=100
emsize=300
nlayers = 1
dropemb = 0.2
droprnn = 0.0
model = RNNModel('GRU', ntokens, emsize, nhidden, 6, nlayers, dropemb=dropemb, droprnn=droprnn, bidirectional=True)
model.encoder.weight.data.copy_(TEXT.vocab.vectors)

import torch.optim as optim
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
if use_cuda:
    model=model.cuda()
    criterion=criterion.cuda()
from tqdm import tqdm_notebook as tqdm

epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_count = 0
    model.train() 
    t = tqdm(train)
    for batch in t:
        (x,xl) = get_text(batch)
        y = get_labels(batch)
        
        optimizer.zero_grad()

        preds = model(x, lengths=xl)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]*len(x)
        running_count += len(x)
        t.set_postfix(loss=(running_loss/running_count))

    epoch_loss = running_loss / running_count

    print('Epoch: {}, Loss: {:.5f}'.format(epoch, epoch_loss))
def get_ids(batch):
    return getattr(batch, id_label).data.cpu().numpy().astype(int)
import numpy as np
test_preds = np.zeros((len(INDEX.vocab), 6))
model.eval()
for batch in test:
    (x,xl) = get_text(batch)
    ids = get_ids(batch)
    preds=model(x,lengths=xl)
    preds = preds.data.cpu().numpy()
    preds = 1/(1+np.exp(-np.clip(preds,-10,10)))
    test_preds[ids]=preds
import pandas as pd
df = pd.read_csv(f'{path}{test_file}')
for i, col in enumerate(label_cols):   
    df[col] = test_preds[1:, i]
df.drop(text_label,axis=1).to_csv("submission.csv",index=False)
df.head(10)
