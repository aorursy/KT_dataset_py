import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.data import Field, BucketIterator, LabelField
from torchtext.datasets import IMDB

import numpy as np
import math
import random
import time
SEED = 1245
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
TEXT = Field(tokenize="spacy")
LABEL = LabelField(dtype=torch.float)

train_data, valid_data = IMDB.splits(TEXT, LABEL)
train_data, test_data = train_data.split(random_state = random.seed(SEED))
print(f"total training examples: {len(train_data)}")
print(f"total test examples: {len(test_data)}")
print(f"total validation examples: {len(valid_data)}")
TEXT.build_vocab(train_data, max_size=25_000)
LABEL.build_vocab(train_data)
print(f"total vocab size {len(TEXT.vocab)}")
print(vars(train_data.examples[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=64, device=device)
class SentimentRNN(nn.Module):
    def __init__(self,input_size, embed_dim, hid_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, input):
        embedded = self.embedding(input)
        out_rnn, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
INPUT_SIZE = len(TEXT.vocab)
EMBED_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = SentimentRNN(INPUT_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
print(model)
def num_parameters(model):
    return sum([params.numel() for params in model.parameters() if params.requires_grad])
print(f"total number of parameters in model : {num_parameters(model):,}")
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3)
# Loss Function
criterion = nn.BCEWithLogitsLoss().to(device)
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc
# Train Model
def train(model,iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for  batch in iterator:
        text = batch.text
        optimizer.zero_grad()
        pred = model(text).squeeze(1)
        loss = criterion(pred, batch.label)
        acc = binary_accuracy(pred, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def validate(model,iterator,criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():        
        for i, batch in enumerate(iterator):
            text, label = batch.text, batch.label
            pred = model(text).squeeze(1)
            loss = criterion(pred,label)
            acc = binary_accuracy(pred, label)
            epoch_loss+= loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
NUM_EPOCH = 5
best_validation_loss = float("inf")
for epoch in range(NUM_EPOCH):
    start_time = time.time()
    training_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    validation_loss, valid_acc = validate(model, valid_iterator, criterion)
    end_time = time.time()
    
    mins, secs = epoch_time(start_time, end_time)
    
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
        
    print(f'Epoch: {epoch+1:02} | Epoch Time: {mins}m {secs}s')
    print(f'\tTrain Loss: {training_loss:.3f} | Train Acc: {train_acc*100:.2f}% %')
    print(f'\t Val. Loss: {validation_loss:.3f} | Train Acc: {valid_acc*100:.2f}% %')
model.load_state_dict(torch.load('tut1-model.pt'))

test_loss, test_acc = validate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
