# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import time

import random

import string



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torchtext import data

from torchtext.vocab import Vectors



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def generate_bigrams(x):

    n_grams = set(zip(*[x[i:] for i in range(2)]))

    for n_gram in n_grams:

        x.append(' '.join(n_gram)) # ["this", 'movie'] -> "this movie"

    return x
generate_bigrams(['this', 'moive', 'is' ,'terrible'])
def load_data(filepath, device):

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, preprocessing=generate_bigrams, fix_length=200)

#     TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=200)

    LABEL = data.Field(sequential=False, use_vocab=False)

    

    field = [('text', TEXT), ('label', LABEL)]

    train, valid, test = data.TabularDataset.splits(filepath, train='Train.csv', validation='Valid.csv', test='Test.csv',

                                                   format='csv', skip_header=True, fields=field)

    cache = '/kaggle/working/vector_cache'

    if not os.path.exists(cache):

        os.mkdir(cache)

    vector = Vectors(name='/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', cache=cache)

    TEXT.build_vocab(train, vectors=vector, max_size=25000, unk_init=torch.Tensor.normal_)

    

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), device=device, batch_size=64, 

                                                             sort_key=lambda x:len(x.text), sort_within_batch=True)

    return TEXT, LABEL, train_iter, valid_iter, test_iter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT, LABEL, train_iter, valid_iter, test_iter = load_data('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format', 

                                                           device=device)
class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx= pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

        

    def forward(self, text):

        # text : [sen_len, batch_size]

        embedded = self.embedding(text)

        # embedded : [sen_len, batch_size, emb_size]

        embedded = embedded.permute(1, 0, 2)

        # embedded : [batch_size, sen_len, emb_size]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled : batch_size,, emb_size

        return self.fc(pooled)

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

OUTPUT_DIM = 1

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



model.embedding.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters())



criterion = nn.BCEWithLogitsLoss()



model = model.to(device)

criterion = criterion.to(device)



def binary_accuracy(preds, y):

    '''

    Returns accuracy per batch...

    '''

    rounded_preds = torch.round(torch.sigmoid(preds)).long()

    correct = (rounded_preds == y).float()

    acc = correct.sum() / len(correct)

    return acc



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time  / 60)

    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))

    return  elapsed_mins, elapsed_secs
def train(model, iterator, optimizer, criterion):

    epoch_loss, epoch_acc = 0, 0

    model.train()

    for i, batch in enumerate(iterator):

        

        predictions = model(batch.text).squeeze()

        

        loss = criterion(predictions, batch.label.float())

        acc = binary_accuracy(predictions, batch.label)

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator) 
def evaluate(model, iterator, criterion):

    

    epoch_loss, epoch_acc = 0, 0

    

    model.eval()

    with torch.no_grad():

        for i, batch in enumerate(iterator):

        

            predictions = model(batch.text).squeeze(1)

        

            loss = criterion(predictions, batch.label.float())

            

            acc = binary_accuracy(predictions, batch.label)

            

            epoch_acc += acc.item()

            epoch_loss += loss.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
N_EPOCHS = 5



best_valid_loss = float('inf')

train_loss_list = []

valid_loss_list = []

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

    train_loss_list.append(train_loss)

    valid_loss_list.append(valid_loss)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'SentimentModel3.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time {epoch_mins}m {epoch_secs}s')

    print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
%pylab inline



plt.figure(figsize=(10, 10))

plt.plot(np.arange(1, N_EPOCHS+1, 1), train_loss_list, 'r', label="Train loss")

plt.plot(np.arange(1, N_EPOCHS+1, 1), valid_loss_list, 'b', label="Valid loss")

plt.xlabel('Epoches')

plt.ylabel('Loss')

plt.grid()
def testModel():

    bestModel = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX).to(device)

    bestModel.load_state_dict(torch.load('SentimentModel3.pt'))

    test_loss, test_acc = evaluate(bestModel, test_iter, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    

testModel()
def predict_sentiment(sentence):

    model.eval()

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    tokenized = generate_bigrams(tokenizer(sentence))

    print(tokenized)

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    print(indexed)

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    bestModel = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX).to(device)

    bestModel.load_state_dict(torch.load('SentimentModel3.pt'))

    prediction = torch.sigmoid(bestModel(tensor))

    return prediction.item()
predict_sentiment("this movie is good, but make me tried")
predict_sentiment("this movie is good")