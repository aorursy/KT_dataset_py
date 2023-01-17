#import libs
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset ,Multi30k
from torchtext.data import Field, Dataset, Iterator, BucketIterator

import numpy as np
import spacy
import random
import math
import time

# Set random seeds for deterministic results 
SEED = 42331
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# install language models
!python -m spacy download en
!python -m spacy download de
# Load language model for basic operations
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def en_tokkenizer(text):
    return [ tok.text for tok in spacy_en.tokenizer(text)]

def de_tokkenizer(text):
    return [ tok.text for tok in spacy_de.tokenizer(text)][::-1] # reversed as mentioned in the paper
# Define source and target text
SRC = Field(init_token="<sos>", eos_token="<eos>", lower=True, tokenize=de_tokkenizer)
TRG = Field(init_token="<sos>", eos_token="<eos>", lower=True, tokenize=en_tokkenizer)
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print(f"Number of traning data {len(train_data.examples)}")
print(f"Number of validation data {len(valid_data.examples)}")
print(f"Number of test data {len(test_data.examples)}")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print(f"Source Vocab list {len(SRC.vocab)}")
print(f"Target Vocab list {len(TRG.vocab)}")
BATCH_SIZE = 125
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator =  BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=DEVICE)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout = dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input):
        embed = self.dropout(self.embeddings(input))
        output, (hidden, cell) = self.lstm(embed)
        return hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.embeddings = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout = dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embed = self.dropout(self.embeddings(input))
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))
        out_prob = self.fc(output.squeeze(0))
        
        return out_prob, hidden, cell
        
        
class Seq2Seq(nn.Module):
    def __init__(self, ecoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_data, trg_data, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(src_data)
        input = trg_data[0,:]
        trg_len = trg_data.shape[0]
        batch_size = trg_data.shape[1]
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)
        for i in range(1, trg_len):
            out_prob, hidden, cell = self.decoder(input, hidden, cell)
            outputs[i] = out_prob
            prediction = out_prob.argmax(1)
            input = trg_data[i] if random.random() < teacher_forcing_ratio else prediction
        return outputs
    
INPUT_SIZE = len(SRC.vocab)
OUTPUT_SIZE = len(TRG.vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5

encoder = Encoder(INPUT_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE)
def init_weight(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weight)

def count_parameters(model):
    count = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return count

print(model)
print(f"Number of parameters {count_parameters(model)}")
optimizer = optim.Adam(model.parameters())
TRG_PAD_INDEX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_INDEX)
def train(iterator, model, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        outputs = model(src,trg)
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(outputs, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+= loss
    return epoch_loss / len(iterator)
        
def validation(iterator, model, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            outputs = model(src,trg, 0)
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(outputs, trg)
            epoch_loss += loss
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(train_iterator,model, optimizer, criterion, CLIP)
    valid_loss = evaluate(valid_iterator, model, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
