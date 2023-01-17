import torch

import random

import numpy as np



def seed_all(seed_value):

    random.seed(seed_value) # Python

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False



seed_all(2020)
!pip install transformers
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_cut(sentence):

    tokens = tokenizer.tokenize(sentence)

    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    tokens = tokens[:max_input_length-2]

    return tokens
from torchtext import data



TEXT = data.Field(batch_first = True,

                  use_vocab = False,

                  tokenize = tokenize_and_cut,

                  preprocessing = tokenizer.convert_tokens_to_ids,

                  init_token = tokenizer.cls_token_id,

                  eos_token = tokenizer.sep_token_id,

                  pad_token = tokenizer.pad_token_id,

                  unk_token = tokenizer.unk_token_id)



LABEL = data.Field(sequential=False, use_vocab=False)
!mkdir ./data

!unzip -d ./data ../input/sentiment-analysis-on-movie-reviews/train.tsv.zip

!unzip -d ./data ../input/sentiment-analysis-on-movie-reviews/test.tsv.zip
from torchtext import datasets

from sklearn.model_selection import train_test_split

import pandas as pd



train = pd.read_csv('./data/train.tsv', sep='\t')

test = pd.read_csv('./data/test.tsv', sep='\t')

train, valid = train_test_split(train, test_size=0.2)

train.to_csv('./data/train.csv', index=False)

valid.to_csv('./data/validation.csv', index=False)



train, valid = data.TabularDataset.splits(

    path='./data', train='train.csv', validation='validation.csv', format='csv', skip_header=True,

    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])

test = data.TabularDataset('./data/test.tsv', format='tsv', skip_header=True,

                           fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])
print(f"Number of training examples: {len(train)}")

print(f"Number of validation examples: {len(valid)}")

print(f"Number of testing examples: {len(test)}")
print(vars(train[6]))
print(tokenizer.convert_ids_to_tokens(vars(train[6])['Phrase']))
BATCH_SIZE = 256



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)

valid_iter = data.BucketIterator(valid, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)

test_iter = data.Iterator(test, batch_size=BATCH_SIZE, train=False, sort=False, device=DEVICE)
batch = next(iter(train_iter))

phrase = batch.Phrase

sent = batch.Sentiment

print(phrase.shape)

print(phrase)

print(sent.shape)

print(sent)
from transformers import BertModel



bert = BertModel.from_pretrained('bert-base-uncased')
from torch import nn



class BERTGRUSentiment(nn.Module):

    def __init__(self, bert, output_dim):

        super().__init__()

        self.bert = bert

        self.embedding_dim = bert.config.to_dict()['hidden_size']

        self.gru11 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)

        self.gru12 = nn.GRU(512, 256, num_layers=1, batch_first=True)

        self.gru13 = nn.GRU(256, 128, num_layers=1, batch_first=True)

        self.gru21 = nn.GRU(self.embedding_dim, 512, num_layers=1, batch_first=True)

        self.gru22 = nn.GRU(512, 256, num_layers=1, batch_first=True)

        self.gru23 = nn.GRU(256, 128, num_layers=1, batch_first=True)



        self.fc = nn.Linear(256, output_dim)

        

        self.dropout1 = nn.Dropout(0.5)

        self.dropout2 = nn.Dropout(0.3)



        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def forward(self, text):

        # text = [batch size, sent len]



        with torch.no_grad():

            embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]



        output1, _ = self.gru11(embedding)

        output1 = self.dropout1(output1)  # output1 = [batch size, sent len, 512]

        

        output1, _ = self.gru12(output1)

        output1 = self.dropout2(output1)  # output1 = [batch size, sent len, 256]

        

        _, hidden1 = self.gru13(output1)  # hidden1 = [1, batch size, 128]



        reversed_embedding = torch.from_numpy(embedding.detach().cpu().numpy()[:, ::-1, :].copy()).to(self.DEVICE)

        

        output2, _ = self.gru21(reversed_embedding)

        output2 = self.dropout1(output2)  # output2 = [batch size, sent len, 512]

        

        output2, _ = self.gru22(output2)

        output2 = self.dropout2(output2)  # output1 = [batch size, sent len, 256]

        

        _, hidden2 = self.gru23(output2)  # hidden2 = [1, batch size, 128]

        

        hidden = self.dropout2(torch.cat((hidden1[-1, :, :], hidden2[-1, :, :]), dim=1))  # hidden = [batch size, 256]



        output = self.fc(hidden)  # output = [batch size, out dim]



        return output
OUTPUT_DIM = 5



model = BERTGRUSentiment(bert, OUTPUT_DIM)
# import torch.nn as nn



# class BERTGRUSentiment(nn.Module):

#     def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

#         super().__init__()

#         self.bert = bert

#         embedding_dim = bert.config.to_dict()['hidden_size']

#         self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional,

#                           batch_first = True, dropout = 0 if n_layers < 2 else dropout)

#         self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

#         self.dropout = nn.Dropout(dropout)

        

#     def forward(self, text):

#         # text = [batch size, sent len]

        

#         with torch.no_grad():

#             embedding = self.bert(text)[0]  # embedding = [batch size, sent len, emb dim]



#         _, hidden = self.gru(embedding)  # hidden = [n layers * n directions, batch size, hid dim]

        

#         if self.gru.bidirectional:

#             hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

#         else:

#             hidden = self.dropout(hidden[-1,:,:])

                

#         # hidden = [batch size, hid dim]

        

#         output = self.fc(hidden)

        

#         # output = [batch size, out dim]

        

#         return output



# HIDDEN_DIM = 256

# OUTPUT_DIM = 5

# N_LAYERS = 2

# BIDIRECTIONAL = True

# DROPOUT = 0.3



# model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
for name, param in model.named_parameters():                

    if name.startswith('bert'):

        param.requires_grad = False



print(f'The model has {count_parameters(model):,} trainable parameters')
for name, param in model.named_parameters():                

    if param.requires_grad:

        print(name)
import torch.optim as optim



optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(DEVICE)

criterion = criterion.to(DEVICE)
import numpy as np



def accuracy(prediction, label):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """

    prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)

    acc = torch.sum(prediction == label).float() / len(prediction == label)

    return acc
def train(model, iterator, optimizer, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        

        optimizer.zero_grad()

        

        data = batch.Phrase

        label = batch.Sentiment

        

        prediction = model(data)

        

        loss = criterion(prediction, label)

        

        acc = accuracy(prediction, label)

        

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

            

            data = batch.Phrase

            label = batch.Sentiment

            

            prediction = model(data)

            

            loss = criterion(prediction, label)

            

            acc = accuracy(prediction, label)



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 10



best_epoch = 0

best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    

    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

        

    end_time = time.time()

        

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        

    if valid_loss < best_valid_loss:

        best_epoch = epoch + 1

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'model.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
from torchtext import data

train_full = data.TabularDataset('./data/train.tsv', format='tsv', skip_header=True,

                                 fields=[('PhraseId', None), ('SentenceId', None),

                                         ('Phrase', TEXT), ('Sentiment', LABEL)])



train_full_iter = data.BucketIterator(train_full, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)



model = BERTGRUSentiment(bert, OUTPUT_DIM)



optimizer = optim.Adam(model.parameters())



criterion = nn.CrossEntropyLoss()



model = model.to(DEVICE)

criterion = criterion.to(DEVICE)



for epoch in range(best_epoch + 1):

    

    start_time = time.time()

    

    train_loss, train_acc = train(model, train_full_iter, optimizer, criterion)

        

    end_time = time.time()

        

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        

    if epoch == best_epoch:

        torch.save(model.state_dict(), 'model_full.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
def predict(model, iterator):

    

    model.eval()

    

    predictions = []

    

    with torch.no_grad():

    

        for batch in iterator:

            

            data = batch.Phrase

            

            prediction = model(data)

            

            prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)

            

            predictions.extend(prediction.tolist())

        

    return predictions
predictions = predict(model, test_iter)



submission = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')

submission['Sentiment'] = predictions

submission.to_csv('submissionBERTGRU.csv', index=False)