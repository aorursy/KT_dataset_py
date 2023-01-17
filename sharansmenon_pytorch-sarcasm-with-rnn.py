# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from torchtext import data

from torchtext import datasets
import torch

from torch import nn, optim
HEADLINE = data.Field(tokenize='spacy', include_lengths = True)

ISSARCASTIC = data.LabelField()
fields = {

    'headline':('headline', HEADLINE),

    'is_sarcastic':('sarcastic', ISSARCASTIC)

}
sarcasm_data = data.TabularDataset(

    path='/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json',

    format='json',

    fields=fields

)
print(vars(sarcasm_data[0]))
train_data, test_data = sarcasm_data.split()
len(test_data)
MAX_VOCAB_SIZE = 50_000
HEADLINE.build_vocab(train_data, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = "glove.6B.100d", 

                 unk_init = torch.Tensor.normal_)
ISSARCASTIC.build_vocab(train_data)
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, test_data), 

    batch_size = BATCH_SIZE,

    sort_key = lambda x:len(x.headline),

    sort_within_batch = True,

    device = device)
class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,

                          dropout=dropout)

        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.fc(hidden)

        
INPUT_DIM = len(HEADLINE.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.5

PAD_IDX = HEADLINE.vocab.stoi[HEADLINE.pad_token]



model = RNN(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = HEADLINE.vocab.vectors



print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = HEADLINE.vocab.stoi[HEADLINE.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
import torch.optim as optim



optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()



model = model.to(device)

criterion = criterion.to(device)
def binary_accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division 

    return correct.sum() / len(correct)
def train(model, iterator, optimizer, criterion):

    epoch_loss = 0.0

    epoch_acc = 0.0

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.headline

        

        predictions = model(text, text_lengths).squeeze(1)

#         print(predictions)

        loss = criterion(predictions, batch.sarcastic.type_as(predictions))

        acc = binary_accuracy(predictions, batch.sarcastic.type_as(predictions))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 5





for epoch in range(N_EPOCHS):



    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    

    end_time = time.time()



    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
def evaluate(model, iterator, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    with torch.no_grad():

    

        for batch in iterator:



            text, text_lengths = batch.headline

            

            predictions = model(text, text_lengths).squeeze(1)

            

            loss = criterion(predictions, batch.sarcastic.type_as(predictions))

            

            acc = binary_accuracy(predictions, batch.sarcastic.type_as(predictions))



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
# model.load_state_dict(torch.load('tut2-model.pt'))



test_loss, test_acc = evaluate(model, test_iterator, criterion)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
import spacy

nlp = spacy.load('en')
def predict_sarcasm(model, sentence):

    model.eval()

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    indexed = [HEADLINE.vocab.stoi[t] for t in tokenized]

    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    length_tensor = torch.LongTensor(length)

    prediction = torch.sigmoid(model(tensor, length_tensor))

    return prediction.item()
predict_sarcasm(model, "She gave him a sarcastic smile.")