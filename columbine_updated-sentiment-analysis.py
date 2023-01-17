# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import time

import string



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import torch

import torch.nn as nn

import torch.optim as optim

from torchtext.vocab import Vectors

from torchtext import data



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_file(file_path, device):

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()



    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, include_lengths=True)

    LABEL = data.Field(sequential=False, use_vocab=False)

    

    datafields = [('text', TEXT), ('label', LABEL)]

    # Step two construction our dataset.

    train, valid, test = data.TabularDataset.splits(path=file_path,

                                                    train="Train.csv", validation="Valid.csv",

                                                    test="Test.csv", format="csv",

                                                    skip_header=True, fields=datafields)

    # because of input dir is read-only we must change the cache path.

    cache = ('/kaggle/working/.vector_cache')

    if not os.path.exists(cache):

        os.mkdir(cache)

    # using the pretrained word embedding.

    vector = Vectors(name='/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', cache=cache)

    TEXT.build_vocab(train, vectors=vector, max_size=25000, unk_init=torch.Tensor.normal_)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), device=device, batch_size=64, 

                                                             sort_key=lambda x:len(x.text), sort_within_batch=True)

    

    return TEXT, LABEL, train_iter, valid_iter, test_iter





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT, LABEL, train_iter, valid_iter, test_iter = load_file('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format', 

                                                          device)
class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)

        

        if bidirectional:

            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        else:

            self.fc = nn.Linear(hidden_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text, text_lengths):

        

        # text : [sen_len, batch_size]

        embedded = self.dropout(self.embedding(text))

        

        # embedded : [sen_len, batch_size, emb_dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        

        # packed_output : [num_word, emb_dim]     hidden : [num_layers * num_direction, batch_size, hid_dim]    

        # cell : [num_layers * num_direction, batch_size, hid_dim]

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        

        #unpacked sequence

        # output : [sen_len, batch_size, hid_dim * num_directions]

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        

        hidden = self.dropout(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)).squeeze()    

        # hidden : [batch_size, hid_dim * num_dir]

        return self.fc(hidden)

    

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)



model.embedding.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters())



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



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time  / 60)

    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))

    return  elapsed_mins, elapsed_secs
def train(model, iterator, optimizer, criterion):

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for i, batch in enumerate(iterator):

        

        text, text_lengths = batch.text

        

        predictions = model(text, text_lengths).squeeze(1)

        

        loss = criterion(predictions, batch.label.float())

        

        acc = binary_accuracy(predictions, batch.label)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        

        if i % 100 == 99:

            print(f"[{i}/{len(iterator)}] : epoch_acc: {epoch_acc / len(iterator):.2f}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            

            text, text_lengths = batch.text

            

            predictions = model(text, text_lengths).squeeze(1)

            

            loss = criterion(predictions, batch.label.float())

        

            acc = binary_accuracy(predictions, batch.label)

            

            epoch_loss += loss.item()

            epoch_acc += acc.item()

            

    return epoch_loss / len(iterator),  epoch_acc / len(iterator)
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



print(f"Test Loss: {test_loss:.3f} | Test Acc : {test_acc*100:.3f}%")
def predict_sentiment(model, sentence):

    model.eval()

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    tokenized = [tok for tok in tokenizer(sentence)]

    print(tokenized)

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    length_tensor = torch.LongTensor(length).to(device)

    prediction = torch.sigmoid(model(tensor, length_tensor))

    return prediction.item()
predict_sentiment(model, "This movie is terrible")
predict_sentiment(model, "This movie is great")