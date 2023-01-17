# import libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

import numpy as np
import time
import random
import spacy
import math
# set random seeds
SEED=1211
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
!python -m spacy download en
!python -m spacy download de

spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokkenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokkenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
# define input and output source fields
SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True, tokenize=tokkenize_de)
TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True, tokenize=tokkenize_en)

# get data from data Source
train_data, valid_data, test_data = Multi30k.splits(exts=('.de','.en'), fields=(SRC,TRG))
print(f"Number of training examples {len(train_data.examples)}")
print(f"Number of valid examples {len(valid_data.examples)}")
print(f"Number of test examples {len(test_data.examples)}")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

BATCH_SIZE=125
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)
class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, en_hidden_dim, out_hidden_dim, dropout_prob):
        super().__init__()
        self.embeddings = nn.Embedding(input_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, en_hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(2* en_hidden_dim, out_hidden_dim)
        
    def forward(self, input):
        embed = self.dropout(self.embeddings(input))
        # enbed size (batch_size, src_len, embed_dim)
        out, hidden = self.rnn(embed)
        # out: ( src_length, batch_size , 2*embed_dim ), hidden: ( 2, batch_size, embed_dim)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        decoder_hidden_init = torch.tanh(self.fc(hidden))
        # decoder_hidden_init: (batch_size, out_hidden_dim)
        return out, decoder_hidden_init
class Attention(nn.Module):
    def __init__(self, en_hidden_dim, de_hidden_dim):
        super().__init__()
        self.attention = nn.Linear(2* en_hidden_dim + de_hidden_dim, de_hidden_dim)
        self.scores = nn.Linear(de_hidden_dim, 1)
        
    def forward(self, en_out, de_hidden):
        src_len = en_out.shape[0]
        de_hidden = de_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # de_hidden: (batch_size, src_len, dec_hidden_dim)
        en_out = en_out.permute(1,0,2)
        energy = torch.tanh(self.attention(torch.cat((en_out, de_hidden), dim=2)))
        # energy: (batch_size, src_len, dec_hidden_dim)
        atten = self.scores(energy).squeeze(2)
        # atten : (batch_size, src_len)
        return F.softmax(atten, dim=1)
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, en_hidden_dim, de_hidden_dim , attention, dropout_prob):
        super().__init__()
        self.output_dim = output_dim
        self.de_hidden_dim = de_hidden_dim
        self.attention = attention
        
        self.embeddings = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim + 2*en_hidden_dim, de_hidden_dim)
        self.fc = nn.Linear(embed_dim + de_hidden_dim + 2*en_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, trg_word ,en_out, de_in_hidden):
        # trg_word : (batch_size), en_out: (src_len, batch_size, 2*en_hidden_dim), de_hidden: (batch_size, de_hidden_dim)
        trg_word = trg_word.unsqueeze(0)
        # trg_word : (1, batch_size)        
        embed = self.dropout(self.embeddings(trg_word))
        # embed : (1, batch_size,embed_dim)
        atten = self.attention(en_out, de_in_hidden)
        en_out = en_out.permute(1,0,2)
        atten = atten.unsqueeze(1)
        # atten : (batch_size,1 ,src_len)
        context_vec = torch.bmm(atten, en_out)
        context_vec = context_vec.permute(1,0,2)
        # context_vec : (1, batch_size, 2*en_hidden_dim)
        rnn_inp = torch.cat((context_vec, embed), dim=2)
        # rnn_inp: (1, batch_size, 2*en_hidden_dim + embed_dim)
        de_out, de_hidden = self.rnn(rnn_inp, de_in_hidden.unsqueeze(0))
        # de_out : (1, batch_size, de_hidden_dim), de_hidden: (1, batch_size, de_hidden_dim)
        de_hidden = de_hidden.squeeze(0)
        context_vec = context_vec.squeeze(0)
        embed = embed.squeeze(0)
        de_out = de_out.squeeze(0)
        de_out = torch.cat((context_vec, de_out, embed), dim=1)
        # de_out: (batch_size, 2*en_hidden_dim + de_hidden_dim + embed_dim)
        out = self.fc(de_out)
        # out: (1, batch_size, output_dim)
        out = out.squeeze(0)
        return out, de_hidden
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_data, trg_data, teacher_forcing_prob=0.5):
        en_out, hidden = self.encoder(src_data)
        batch_size = trg_data.shape[1]
        trg_len = trg_data.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)
        input = trg_data[0,:]
        for i in range(1, trg_len):
            out_prob, hidden = self.decoder(input, en_out, hidden)
            outputs[i] = out_prob
            pred = out_prob.argmax(1)
            input = trg_data[i] if random.random() < teacher_forcing_prob else pred
        return outputs
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attn ,DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)
def init_weight(model):
    for name, params in model.named_parameters():
        nn.init.normal_(params.data, mean=0, std=0.01)
model.apply(init_weight)

def count_params(model):
    return sum([params.numel() for params in model.parameters() if params.requires_grad])

print(model)
print(f"Number of parameters in model : {count_params(model):,}")

# Define Optimizer and Loss function
optimizer = optim.Adam(model.parameters())
pad_inx = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_inx)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def train(iterator, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        outputs = model(src,trg)
        vocab_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, vocab_dim)
        trg = trg[1:].view(-1)
        loss = criterion(outputs,trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss
    return epoch_loss/len(iterator)
def evaluate(iterator,model,criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            outputs = model(src,trg)
            vocab_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, vocab_dim)
            trg = trg[1:].view(-1)
            loss = criterion(outputs,trg)
            epoch_loss += loss
    return epoch_loss / len(iterator)
EPOCH_SIZE = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(EPOCH_SIZE):

    start_time = time.time()    
    train_loss = train(train_iterator,model, optimizer, criterion, CLIP)
    valid_loss = evaluate(valid_iterator, model, criterion)    
    end_time = time.time()

    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')    

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss = evaluate(test_iterator, model , criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
