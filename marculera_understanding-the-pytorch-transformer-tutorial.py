import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src.cuda()
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        self.src_mask = self.src_mask.cuda()
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)    
    
    def generate_sequence(self, sequence, seq_size=2):
        sequence = sequence.unsqueeze(1)
        generate_step = 0
        while generate_step < seq_size:
          output_word = torch.argmax(self.forward(sequence)[-1, :], dim=1).unsqueeze(0)
          sequence = torch.cat((sequence, output_word), dim=0)
          generate_step += 1
        sequence = sequence.squeeze(1)
        return sequence

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
import pandas as pd
dummy_dataset = pd.read_csv('../input/dummylanguage/train.tsv', sep="\t", header=None)
dummy_dataset.head()
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import LanguageModelingDataset
from torchtext.data import Example, RawField, BucketIterator, BPTTIterator

DUMMY_TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>')
my_dataset = LanguageModelingDataset("../input/dummylanguage/train.tsv", DUMMY_TEXT)
DUMMY_TEXT.build_vocab(my_dataset)
train_iter = BPTTIterator(my_dataset, batch_size=3, bptt_len=4)
def words_in_sequence(seq, torch_text_field):
    seq_list = seq.tolist()
    words = [torch_text_field.vocab.itos[word_idx] for word_idx in seq_list]
    return ' '.join(words)

def print_train_target_iterator(bptt_iterator, torch_text_field, only_first_batch=False):
    for i in bptt_iterator:
        for (train, target) in zip(i.text, i.target): 
            print(f'train: {words_in_sequence(train, torch_text_field)}')
            print(f'target: {words_in_sequence(target, torch_text_field)}')
            print()
        if only_first_batch:
            break
print_train_target_iterator(train_iter, DUMMY_TEXT)

ntokens = len(DUMMY_TEXT.vocab.stoi) # the size of vocabulary
emsize = 20 # embedding dimension
nhid = 1 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
print(model)
for i in train_iter:
    print(f'Model input dimensions: {i.text.shape}')
    break
print(f'Model output dimensions: {model(i.text).shape}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
model.train() 
for j in range(100):
    for i in train_iter:
        optimizer.zero_grad()
        output = model(i.text)
        loss = criterion(output.view(-1, output.shape[-1]), i.target.view(-1).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        print(f'Loss: {loss.item()}')
    
def predict_sequence(source_sentence, torch_text_field, model, seq_size=2):
    print(f"Source: {' '.join(source_sentence)}")
    generated_sequence = model.generate_sequence(torch_text_field.numericalize([source_sentence]).to(device).squeeze(1),
                                                 seq_size)
    print(f'Result: {words_in_sequence(generated_sequence, torch_text_field)}')
model.eval()

predict_sequence(["abc","def"], DUMMY_TEXT, model)
predict_sequence(["lmn","opq"], DUMMY_TEXT, model, 3)

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
train_iter = BPTTIterator(train_txt, batch_size=20, bptt_len=35)
ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
print(model)
print_train_target_iterator(train_iter, TEXT, True)

for i in train_iter:
    print(f'Model input dimensions: {i.text.shape}')
    break
print(f'Model output dimensions: {model(i.text).shape}')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
bptt = 35 

import time
def train():
    model.train() 
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(train_iter):
        data = i.text
        targets = i.target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.shape[-1]), i.target.view(-1).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    
    eval_model.eval() 
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
import time
best_val_loss = float("inf")
epochs = 4 
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
model.eval()

predict_sequence(["american","football"], TEXT, model)


predict_sequence(["conrad","said"], TEXT, model, 7)
predict_sequence(["players","play"], TEXT, model, 4)