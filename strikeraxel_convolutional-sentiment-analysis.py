import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets

import numpy as np
import math
import random
import time
SEED = 12312
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# Download Train and test datasets
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float32)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(f"Size of train dataset : {len(train_data)}")
print(f"Size of validation dataset : {len(valid_data)}")
print(f"Size of test dataset : {len(test_data)}")
# Create vocabulary
MAX_VOCAB = 25_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB, vectors = "glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# Create batches for training, validation and test datasets
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), 
                                                                                batch_size=BATCH_SIZE,
                                                                                device=device
                                                                              )
class SentimentCNN(nn.Module):
    def __init__(self, vocab_dim, embed_dim, num_features ,filters,output_dim, pad_idx):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim, padding_idx=pad_idx )
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_features,
                      kernel_size=fil
                     )
            for fil in filters
        ])
        self.fc = nn.Linear(len(filters)*num_features, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, txt):
        embed = self.embeddings(txt)
        embed = embed.permute(1,2,0)
        conv_features = [ F.relu(conv(embed)) for conv in self.convs]
        conv_features = [ F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_features]
        out = self.dropout(torch.cat(conv_features, dim=1))
        out = self.fc(out)
        return out
EMBED_SIZE = 100
VOCAB_SIZE = len(TEXT.vocab)
OUT_DIM = 1
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = SentimentCNN(VOCAB_SIZE, EMBED_SIZE, N_FILTERS, FILTER_SIZES , OUT_DIM, PAD_IDX)

def num_params(model):
    return sum([params.numel() for params in model.parameters() if params.requires_grad])

print(model)
print(f"Number of parameters in model : {num_params(model):,}")
pretrained_vectors = TEXT.vocab.vectors
model.embeddings.weight.data.copy_(pretrained_vectors)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embeddings.weight.data[UNK_IDX] = torch.zeros(EMBED_SIZE)
model.embeddings.weight.data[PAD_IDX] = torch.zeros(EMBED_SIZE)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model.to(device)
criterion.to(device)
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
