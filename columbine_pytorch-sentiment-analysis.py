# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import string 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.optim as optim

from torchtext import data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_file(filepath, device, MAX_VOCAB_SIZE = 25_000):

    # our tokenizer: removing the punctuation & spliting the sentence.

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    

    # Step one defination of our fields. 

    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=100)

    LABEL = data.Field(sequential=False, use_vocab=False)

    

    print("loading from csv ...")

    tv_datafields = [("text", TEXT), ("label", LABEL)]

    

    # Step two construction our dataset.

    train, valid, test = data.TabularDataset.splits(path=filepath,

                                                    train="Train.csv", validation="Valid.csv",

                                                    test="Test.csv", format="csv",

                                                    skip_header=True, fields=tv_datafields)

    print(train[0].__dict__.keys())

    

    

    # Step three We should build_vocab for the field with use_vocab=True. 

    # If not we will get an error during the loop section.

    TEXT.build_vocab(train, max_size = MAX_VOCAB_SIZE)

    

    print("build vocab success...")

    

    # Step four construct our iterator to our dataset. 

    train_iter = data.BucketIterator(train, device=device, batch_size=32, sort_key=lambda x: len(x.text),

                                     sort_within_batch=False, repeat=False)

    valid_iter = data.BucketIterator(valid, device=device, batch_size=32, sort_key=lambda x: len(x.text),

                                     sort_within_batch=False, repeat=False)

    test_iter = data.BucketIterator(test, device=device, batch_size=32, sort_key=lambda x: len(x.text),

                                     sort_within_batch=False, repeat=False)

    print("construct iterator success...")

    return TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter = load_file('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format', device)
# most common words and their frequencies.

print(TEXT.vocab.freqs.most_common(20))



# top ten index to words transform.

print(TEXT.vocab.itos[:10])
class SentimentModel(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        

    def forward(self, text):

        # text [sentence length, batch_size]



        embedded = self.embedding(text)

        

        # embedded = [sentence length, batch_size, emb dim]

        output, hidden = self.rnn(embedded)

        

        # output = [sent len, batch_size, hid dim]

        # hidden = [1, batch_size, hid dim]

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        

        return self.fc(hidden.squeeze(0))   
INPUT_DIM = len(TEXT.vocab)

print(INPUT_DIM)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1



model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)

criterion = criterion.to(device)
def binary_accuracy(preds, y):

    '''

    Return accuracy per batch ..

    '''

    

    # round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float()

    acc = correct.sum() / len(correct)

    

    return acc
def train(model, iterator, optimizer, criterion):

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()



        predictions = model(batch.text).squeeze(1)

        

        # note we must transform the batch.label into float or we will get an error later.

        loss = criterion(predictions, batch.label.float())

        acc = binary_accuracy(predictions, batch.label)

        

        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        if i % 200 == 199:

            print(f"[{i}/{len(iterator)}] : epoch_acc: {epoch_acc / len(iterator):.2f}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            # prediction [batch_size]

            predictions = model(batch.text).squeeze(1)

            

            loss = criterion(predictions, batch.label.float())

            

            acc = binary_accuracy(predictions, batch.label)

        

            epoch_loss += loss.item()

            epoch_acc += acc.item()

            

    return epoch_loss / len(iterator),  epoch_acc / len(iterator)
import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time  / 60)

    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))

    return  elapsed_mins, elapsed_secs
N_epoches = 5



best_valid_loss = float('inf')



for epoch in range(N_epoches):

    

    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

    

    end_time = time.time()

    

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'Sentiment-model.pt')

        

    print(f'Epoch:  {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain  Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\tValid  Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')
model.load_state_dict(torch.load('Sentiment-model.pt'))



test_loss, test_acc = evaluate(model, test_iter, criterion)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')