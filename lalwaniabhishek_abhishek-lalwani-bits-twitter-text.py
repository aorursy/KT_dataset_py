#Changing the 4 to 1

import pandas as pd

train_with_4 = pd.read_csv('../input/train.csv')

train_with_4['target'] = train_with_4['target']/4

train_with_4.to_csv('train.csv', index = False)
#Testing the change

import pandas as pd

train_with_1 = pd.read_csv('train.csv')

train_with_1.head()
#Defining the generate bigrams method for the Fast_Text class

def generate_bigrams(x):

    n_grams = set(zip(*[x[i:] for i in range(2)]))

    for n_gram in n_grams:

        x.append(' '.join(n_gram))

    return x
#Testing the method

generate_bigrams(['This', 'film', 'is', 'terrible'])
#Getting the relevant imports and the fields for reading training data

import torch

from torchtext import data

from torchtext import datasets

import random

import pandas as pd

import numpy as np



SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



TEXT = data.Field(preprocessing = generate_bigrams)

TARGET = data.LabelField(dtype = torch.float)
#Defining the fields for reading train.csv

fields_train = [(None, None), (None, None), (None, None), ('text', TEXT),('target', TARGET)]
#Reading train.csv

train_data = data.TabularDataset(path = 'train.csv',

                                 format = 'csv',

                                 fields = fields_train,

                                 skip_header = True

)
#Testing whether the data was imported successfully

print(vars(train_data[0]))
#Creating validation set

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
#Getting the pre-trained word embeddings and building the vocab

MAX_VOCAB_SIZE = 25_000



TEXT.build_vocab(train_data, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = "glove.6B.100d", 

                 unk_init = torch.Tensor.normal_)



TARGET.build_vocab(train_data)
#defining the batch size and defining the iterators for training and validation data

BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator = data.Iterator(dataset = train_data, batch_size = BATCH_SIZE,device = device, shuffle = None, train = True, sort_key = lambda x: len(x.text), sort_within_batch = False)

valid_iterator = data.Iterator(dataset = valid_data, batch_size = BATCH_SIZE,device = device, shuffle = None, train = False, sort_key = lambda x: len(x.text), sort_within_batch = False)
#defining the Fast_Text Class

import torch.nn as nn

import torch.nn.functional as F



class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        

        self.fc = nn.Linear(embedding_dim, output_dim)

        

    def forward(self, text):

        

        #text = [sent len, batch size]

        

        embedded = self.embedding(text)

                

        #embedded = [sent len, batch size, emb dim]

        

        embedded = embedded.permute(1, 0, 2)

        

        #embedded = [batch size, sent len, emb dim]

        

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 

        

        #pooled = [batch size, embedding_dim]

                

        return self.fc(pooled)
#defining our models and the relevant parameters

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

OUTPUT_DIM = 1

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
#counting the number of parameters which our model will process

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
#Copying the pre-trained vectors to our embedding layers

pretrained_embeddings = TEXT.vocab.vectors



model.embedding.weight.data.copy_(pretrained_embeddings)
#Zeroing the initial weight of our unknown and padding tokens

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#defining our optimizer

import torch.optim as optim



optimizer = optim.Adam(model.parameters())
#defining our loss and porting our model and loss to GPU

criterion = nn.BCEWithLogitsLoss()



model = model.to(device)

criterion = criterion.to(device)
#defining the accuracy calculation method

def binary_accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division 

    acc = correct.sum() / len(correct)

    return acc
#defining the training method

def train(model, iterator, optimizer, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        

        optimizer.zero_grad()

        

        predictions = model(batch.text).squeeze(1)

        

        loss = criterion(predictions, batch.target)

        

        acc = binary_accuracy(predictions, batch.target)

        

        loss.backward()

        

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
#defining the validation method

def evaluate(model, iterator, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    with torch.no_grad():

    

        for batch in iterator:



            predictions = model(batch.text).squeeze(1)

            

            loss = criterion(predictions, batch.target)

            

            acc = binary_accuracy(predictions, batch.target)



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
#defining the method to calculate epoch time

import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
#TRAINING!

N_EPOCHS = 20



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
#Inference method

def predict_sentiment(model, sentence):

    model.eval()

    tokenized = generate_bigrams(sentence.split())

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    prediction = torch.sigmoid(model(tensor))

    return prediction.item()
#Running inference on test

preds = []

test_data = pd.read_csv('../input/test.csv')

for i in range(len(test_data)):

    preds.append((int(predict_sentiment(model, test_data['text'][i])>0.5))*4)

ids = test_data['Id']

dict = {'Id': ids, 'target': preds}

df = pd.DataFrame(dict) 



df.to_csv('final_output.csv', index=False)
#Checking whether the result file is created properly

test_data = pd.read_csv('final_output.csv')

test_data.head()