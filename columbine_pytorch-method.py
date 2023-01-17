# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import time

import string



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

from torchtext import data

import torch.optim as optim

from torchtext.vocab import Vectors

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_data(file_path, device):

    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    TEXT = data.Field(sequential=True, lower=True, include_lengths=True, tokenize=tokenizer)

    LABEL = data.Field(sequential=False, use_vocab=False)

    

    trn_dataField = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]

    tst_dataField = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)]

    

    train = data.TabularDataset(path=os.path.join(file_path, 'train.tsv'), format='tsv', skip_header=True, fields=trn_dataField)

    test = data.TabularDataset(path=os.path.join(file_path, 'test.tsv'), format='tsv', skip_header=True, fields=tst_dataField)

    

    train, valid = train.split(random_state=random.seed(1234))

    cache = ('/kaggle/working/.vector_cache')

    if not os.path.exists(cache):

        os.mkdir(cache)

    # using the pretrained word embedding.

    vector = Vectors(name='/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', cache=cache)

    TEXT.build_vocab(train, vectors=vector, unk_init=torch.Tensor.normal_)

    

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), device=device, batch_size=64, 

                                                       sort_key=lambda x:len(x.Phrase), sort_within_batch=True)

    return TEXT, LABEL, train_iter, valid_iter, test_iter





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT, LABEL, train_iter, valid_iter, test_iter = load_data('/kaggle/input/sentiment-analysis-on-movie-reviews', device)
class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)

        

        self.rnn = nn.LSTM(embedding_size, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)

        

        if bidirectional:

            self.fc = nn.Linear(2 * hidden_dim, output_dim)

        else:

            self.fc = nn.Linear(hidden_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

    

    def forward(self, text, lengths):

        

        embedded = self.embedding(text)   #embedded : [sen_len, batch_size, emb_dim]

        

        packed_embedded = pack_padded_sequence(embedded, lengths)

        

        # packed_output : [num_word, emb_dim]     hidden : [num_layers * num_direction, batch_size, hid_dim]    

        # cell : [num_layers * num_direction, batch_size, hid_dim]

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        

        output, output_length = pad_packed_sequence(packed_output)

        

        hidden = self.dropout(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)).squeeze()

         # hidden : [batch_size, hid_dim * num_dir]

        return self.fc(hidden)

    

    

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 5

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



criterion = nn.CrossEntropyLoss()



model = model.to(device)

criterion = criterion.to(device)
a = torch.Tensor([[1,1,2,1], [1,2,3,4]])

print(a.shape)

b = torch.softmax(a, dim=1)

print(torch.argmax(b, dim=1))
def accuracy(preds, y):

    '''

    Return accuracy per batch ..

    '''

    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)

    correct = (preds == y).float()

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

        

        text, text_lengths = batch.Phrase

        

        if(torch.min(text_lengths) <= 0): 

            continue



        predictions = model(text, text_lengths)

        

        loss = criterion(predictions, batch.Sentiment)

        

        acc = accuracy(predictions, batch.Sentiment)

        

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

            

            text, text_lengths = batch.Phrase

            

            if(torch.min(text_lengths) <= 0): 

#                 continue

                predictions = torch.Tensor([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],

                                           [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]).to(device)

            else:

                predictions = model(text, text_lengths)

            

            loss = criterion(predictions, batch.Sentiment)

        

            acc = accuracy(predictions, batch.Sentiment)

            

            epoch_loss += loss.item()

            epoch_acc += acc.item()

            

    return epoch_loss / len(iterator),  epoch_acc / len(iterator)
N_epoches = 10



trainLossRecords = []

validLossRecords = []



best_valid_loss = float('inf')



for epoch in range(N_epoches):

    

    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)

    # get the loss records to visualize.

    trainLossRecords.append(train_loss)

    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

    validLossRecords.append(valid_loss)

    end_time = time.time()

    

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if train_loss < best_valid_loss:

        best_valid_loss = train_loss

        torch.save(model.state_dict(), 'Sentiment-model.pt')

        

    print(f'Epoch:  {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain  Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\tValid  Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')
%pylab inline

epoches = np.arange(1, N_epoches + 1, 1)

plt.figure(figsize=(10, 10))

plt.title('Train & Valid Loss')

plt.xlabel(r'Epoch')

plt.ylabel(r'Loss')

plt.plot(epoches, trainLossRecords, 'r.', label='Train Loss')

plt.plot(epoches, validLossRecords, 'b.', label='Valid loss')

plt.grid()

def Submission():

    prediction = torch.Tensor([]).to(device)

    

    # load our best model parameters.

    best_model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    best_model.load_state_dict(torch.load('Sentiment-model.pt'))

    best_model.eval()

    best_model.to(device)

    

    # get the prediction 

    for i, batch in enumerate(test_iter):

        text, text_lengths = batch.Phrase

        if(torch.min(text_lengths) <= 0): 

            batch_predict = torch.Tensor([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

                                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

                                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

                                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,]).to(device)

        else:

            batch_predict = best_model(text, text_lengths)

            batch_predict = torch.argmax(torch.softmax(batch_predict, dim=1), dim=1)

        prediction = torch.cat([prediction, batch_predict.float()], dim=0)

    print(prediction[10000:10100])

    # submission our results.

    sub_file = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv',sep=',')

    sub_file.Sentiment=prediction.cpu().numpy().astype(int).tolist()



    sub_file.to_csv('Submission.csv', index=False)



Submission()