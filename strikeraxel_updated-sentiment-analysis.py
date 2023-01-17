import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import IMDB

import numpy as np
import time
import math
import random
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = Field(tokenize='spacy', include_lengths=True)
LABEL = LabelField(dtype=torch.float)
train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))
print(f"Number of data in training example : {len(train_data)}")
print(f"Number of data in validation example : {len(valid_data)}")
print(f"Number of data in test example : {len(test_data)}")
vars(train_data.examples[0])
MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors='glove.6B.100d',unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
print(len(TEXT.vocab))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                               batch_size = 64,
                                                               sort_within_batch = True,
                                                               device = device)
class SentimentRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers, bi_directional  ,output_dim , dropout_prob , pad_inx ):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx = pad_inx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional = bi_directional, dropout = dropout_prob)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, src_txt, src_length):
        embed = self.dropout(self.embedding(src_txt))
        padd_embed = nn.utils.rnn.pack_padded_sequence(embed, src_length)
        output, hidden = self.rnn(padd_embed)
        cat_hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(cat_hidden)
INPUT_SIZE = len(TEXT.vocab)
OUT_SIZE = 1
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BI_DIRECTIONAL = True
DROPOUT_PROB = 0.5
CLIP = 1

PAD_INX = TEXT.vocab.stoi[TEXT.pad_token]
model = SentimentRNN(INPUT_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BI_DIRECTIONAL, OUT_SIZE, DROPOUT_PROB, PAD_INX).to(device)
print(model)
def num_params(model):
    return sum([params.numel() for params in  model.parameters() if params.requires_grad])
print(f"Number of parameters in model : {num_params(model):,}")
# Init weight embeddings 
UNK_INX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.embedding.weight.data[PAD_INX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[UNK_INX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)
def acc_prediction(pred, y):
    rounded_preds = torch.round(torch.sigmoid(pred))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc
def train(model, iterator, criterion, optimizer, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        text, text_lengths = batch.text
        label = batch.label
        optimizer.zero_grad()
        pred = model(text, text_lengths).squeeze(1)
        loss = criterion(pred, label)
        acc = acc_prediction(pred, label)
        loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()        
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def validate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            label = batch.label
            pred = model(text, text_lengths).squeeze(1)
            loss = criterion(pred, label)
            acc = acc_prediction(pred, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()        
    return epoch_loss/len(iterator), epoch_acc/len(iterator)            
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
NUM_EPOCH=10
best_loss = float("inf")
for epoch in range(NUM_EPOCH):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, criterion, optimizer,CLIP)
    valid_loss, valid_acc = validate(model, valid_iterator, criterion)
    end_tim = time.time()
    mins, secs = epoch_time(start_time, end_tim)
    
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
        
    print(f"Epoch : {epoch + 1:02}, Epoch Time: {mins}s : {secs}s ")
    print(f"\t Train loss : {train_loss:.3f} | Train Acc: {train_acc*100:.2f}")
    print(f"\t Valid loss : {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}")
model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc = validate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

