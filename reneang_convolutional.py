# My solution based on the great tutorial:

# https://github.com/bentrevett/pytorch-sentiment-analysis



%matplotlib inline

import numpy as np 

import pandas as pd 

import torch

import torchtext

from torchtext import data

import spacy

import os

import re





os.environ['OMP_NUM_THREADS'] = '4'





SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



TEXT = data.Field(lower=True,include_lengths=True ,tokenize='spacy')



LABEL = data.Field(sequential=False, 

                         use_vocab=False, 

                         pad_token=None, 

                            unk_token=None, dtype = torch.float)









dataFields = {"comment_text": ("comment_text", TEXT), 

              'toxic': ("toxic", LABEL), 

              'severe_toxic': ("severe_toxic", LABEL),

              'threat': ("threat", LABEL), 

              'obscene': ("obscene", LABEL),

              'insult': ("insult", LABEL), 

              'identity_hate': ("identity_hate", LABEL)}



dataset= data.TabularDataset(path='../input/toxicjson/train.json', 

                                            format='json',

                                            fields=dataFields, 

                                            skip_header=True)
import random

SEED = 3

#train, unimportant = dataset.split(split_ratio=0.5,random_state = random.seed(SEED)) 



train_data, val_data = dataset.split(split_ratio=0.5,random_state = random.seed(SEED))
MAX_VOCAB_SIZE = 20_000



TEXT.build_vocab(train_data, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = "glove.6B.100d", 

                 unk_init = torch.Tensor.normal_)
BATCH_SIZE = 512



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_data, val_data), 

    batch_size = BATCH_SIZE,

    sort_key=lambda x: len(x.comment_text),

    sort_within_batch = True,

    device = device)
yFields = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

iaux=0

for batch in valid_iterator:

    iaux+=1

    aux = batch

    aux2= torch.stack([getattr(batch, y) for y in yFields])

    if iaux==20: break


        

torch.transpose( torch.stack([getattr(aux, y) for y in yFields]),0,1)
aux.comment_text[0].size()
aux.toxic.size()
import torch.nn as nn

from torch.functional import F

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 

                 dropout, pad_idx):

        

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        

        self.convs = nn.ModuleList([

                                    nn.Conv2d(in_channels = 1, 

                                              out_channels = n_filters, 

                                              kernel_size = (fs, embedding_dim)) 

                                    for fs in filter_sizes

                                    ])

        

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text):

        

        #text = [sent len, batch size]

        

        text = text.permute(1, 0)

                

        #text = [batch size, sent len]

        

        embedded = self.embedding(text)

                

        #embedded = [batch size, sent len, emb dim]

        

        embedded = embedded.unsqueeze(1)

        

        #embedded = [batch size, 1, sent len, emb dim]

        

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

            

        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        

        #pooled_n = [batch size, n_filters]

        

        cat = self.dropout(torch.cat(pooled, dim = 1))



        #cat = [batch size, n_filters * len(filter_sizes)]

            

        return self.fc(cat)





INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

N_FILTERS = 100

FILTER_SIZES = [3,3,4]

OUTPUT_DIM = 6

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
import torch.optim as optim



optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()



model = model.to(device)

criterion = criterion.to(device)
import numpy

from sklearn.metrics import roc_auc_score

def roc_auc(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """

    #round predictions to the closest integer

    #rounded_preds = torch.sigmoid(preds)

    

    #assert preds.size()==y.size()

    

    #reds=rounded_preds.detach().numpy()



    #y=y.numpy()

    

    global var_y

    global var_preds

    var_y = y

    var_preds = preds

    print('jeje', y.shape)

    acc = roc_auc_score(y, preds)

    print('jojo',preds.shape)

    

    return acc




def train(model, iterator, optimizer, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    preds_list=[]

    labels_list= []

 

    

    for i, batch in enumerate(iterator):

        

        optimizer.zero_grad()

        

        text, text_lengths = batch.comment_text

        

        predictions = model(text).squeeze(1)

        

        batch_labels=torch.stack([getattr(batch, y) for y in yFields]) #transpose?

        batch_labels = torch.transpose(batch_labels,0,1)

        

        loss = criterion(predictions, batch_labels)

        

        loss.backward()

        

        optimizer.step()

        

        preds_list+=[torch.sigmoid(predictions).detach().numpy()]

        labels_list+=[batch_labels.numpy()]

        

        #if i%64==0:

        #    epoch_acc += [roc_auc(np.vstack(preds_list), np.vstack(batch_labels))]

        #    preds_list=[]

        #    labels_list= []

            

        

        epoch_loss += loss.item()

        

        

        

    return epoch_loss / len(iterator), roc_auc(np.vstack(preds_list), np.vstack(labels_list))




def evaluate(model, iterator, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    preds_list=[]

    labels_list= []

    epoch_acc=[]

    

    with torch.no_grad():

    

        for batch in iterator:



            text, text_lengths = batch.comment_text

            

            predictions = model(text).squeeze(1)

            

            batch_labels = torch.stack([getattr(batch, y) for y in yFields]) #transpose?

            batch_labels = torch.transpose(batch_labels,0,1)

            

            loss = criterion(predictions, batch_labels)



            epoch_loss += loss.item()

            

            preds_list+=[torch.sigmoid(predictions).detach().numpy()]

            labels_list+=[batch_labels.numpy()]

        

            #if i%64==0:

            #    epoch_acc += [roc_auc(np.vstack(preds_list), np.vstack(batch_labels))]

            #    preds_list=[]

            #    labels_list= []

        

    return epoch_loss / len(iterator), roc_auc(np.vstack(preds_list), np.vstack(labels_list))



from torchsummary import summary
import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs




N_EPOCHS = 8



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):



    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    print('jaja')

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    print('juju')

    end_time = time.time()



    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'tut2-model.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')



test_dataFields = {"comment_text": ("comment_text", TEXT)}



test_dataset= data.TabularDataset(path='./data/test.json', 

                                            format='json',

                                            fields=test_dataFields, 

                                            skip_header=True)
test_iterator = data.BucketIterator.splits(test_dataset, device = device,

    batch_size = BATCH_SIZE, sort=False, sort_key=lambda x: len(x.comment_text), 

    sort_within_batch=False, repeat=False, shuffle=False)
myPreds=[]

with torch.no_grad():

    model.eval()

    for batch in test_iterator:



        torch.cuda.empty_cache()

    

        text, text_lengths = batch.comment_text    

        predictions = model(text).squeeze(1)         

        myPreds+=[torch.sigmoid(predictions).detach().numpy()]

    

        torch.cuda.empty_cache()

myPreds = np.vstack(myPreds)
test_dataset[0].comment_text
for batch in test_iterator:

    aux3= batch

    break
aux3
dataFields = {"comment_text": ("comment_text", TEXT)}



testDataset= data.TabularDataset(path='../input/toxicjson/test.json', 

                                            format='json',

                                            fields=dataFields, 

                                            skip_header=False)
len(testDataset)
test_iterator = torchtext.data.Iterator(testDataset, batch_size=32, device=device, 

                                     sort=False, sort_within_batch=False, 

                                     repeat=False,shuffle=False)
myPreds=[]

with torch.no_grad():

    model.eval()

    for batch in test_iterator:



        torch.cuda.empty_cache()

    

        text, text_lengths = batch.comment_text    

        predictions = model(text).squeeze(1)         

        myPreds+=[torch.sigmoid(predictions).detach().numpy()]

    

        torch.cuda.empty_cache()

myPreds = np.vstack(myPreds)
testDF= pd.read_csv("../input/test.csv")

for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):

    testDF[col] = myPreds[:, i]
myPreds.shape
testDF.drop("comment_text", axis=1).to_csv("submission.csv", index=False)