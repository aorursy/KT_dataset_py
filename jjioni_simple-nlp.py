# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import time



import torch

from torchtext import data

import torch.nn as nn
# Import Data



test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
test.describe()
len(test)
test['text']
# Shape of dataset

train.shape
train.head()
# drop 'id' , 'keyword' and 'location' columns.

train.drop(columns=['id','keyword','location'], inplace=True)
# to clean data

def normalise_text (text):

    text = text.str.lower() # lowercase

    text = text.str.replace(r"\#","") # replaces hashtags

    text = text.str.replace(r"http\S+","URL")  # remove URL addresses

    text = text.str.replace(r"@","")

    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")

    text = text.str.replace("\s{2,}", " ")

    return text
train["text"]=normalise_text(train["text"])
train['text'].head()
# split data into train and validation 

train_df, valid_df = train_test_split(train)
train_df.head()
valid_df.head()
train_df.shape
SEED = 42



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
TEXT = data.Field(tokenize = 'spacy', include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)
class DataFrameDataset(data.Dataset):



    def __init__(self, df, fields, is_test=False, **kwargs):

        examples = []

        for i, row in df.iterrows():

            label = row.target if not is_test else None

            text = row.text

            examples.append(data.Example.fromlist([text, label], fields))



        super().__init__(examples, fields, **kwargs)



    @staticmethod

    def sort_key(ex):

        return len(ex.text)



    @classmethod

    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):

        train_data, val_data, test_data = (None, None, None)

        data_field = fields



        if train_df is not None:

            train_data = cls(train_df.copy(), data_field, **kwargs)

        if val_df is not None:

            val_data = cls(val_df.copy(), data_field, **kwargs)

        if test_df is not None:

            test_data = cls(test_df.copy(), data_field, True, **kwargs)



        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
fields = [('text',TEXT), ('label',LABEL)]



train_ds, val_ds= DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)



test_ds = data.TabularDataset(

    path='../input/nlp-getting-started/test.csv', format='csv',

    skip_header=True,

    fields=[

        ('text', TEXT)

    ]

)
len(test_ds)
def get_iterator(dataset, batch_size, train=True,

                 shuffle=True, repeat=False):

    

    device = torch.device('cpu')

    

    dataset_iter = data.Iterator(

        dataset, batch_size=batch_size, device=device,

        train=train, shuffle=shuffle, repeat=repeat,

        sort=False

    )

    

    return dataset_iter
# Lets look at a random example

print(vars(test_ds[15]))



# Check the type 

print(type(test_ds[15]))
MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(train_ds, test_ds,

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = 'glove.6B.200d',

                 unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds)
print(LABEL.vocab.stoi)
print(TEXT.vocab.stoi)
print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))
#No. of unique tokens in text

print("Size of TEXT vocabulary:",len(TEXT.vocab))



#No. of unique tokens in label

print("Size of LABEL vocabulary:",len(LABEL.vocab))



#Commonly used words

print(TEXT.vocab.freqs.most_common(10))  



#Word dictionary

# print(TEXT.vocab.stoi)   
print(TEXT.vocab.vectors[1].shape) # <unk>의 임베딩 벡터값
print(TEXT.vocab.vectors[255]) # <unk>의 임베딩 벡터값
BATCH_SIZE = 128



#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

print(device)



from torchtext.data import Field, BucketIterator



train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_ds, val_ds),

    batch_size = BATCH_SIZE,

    device = device)



test_iter = get_iterator(test_ds, batch_size=1, 

                         train=False, shuffle=False,

                         repeat=False)

'''

train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_ds, val_ds), 

    batch_size = BATCH_SIZE,

    sort_within_batch = True,

    device = device)

'''

len(test_iter)
from tqdm.notebook import tqdm

for batch in tqdm(test_iter):

    print(batch)
# Hyperparameters



num_epochs = 25

learning_rate = 0.001



INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 200

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.2

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
class LSTM_net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.rnn = nn.LSTM(embedding_dim, 

                           hidden_dim, 

                           num_layers=n_layers, 

                           bidirectional=bidirectional, 

                           dropout=dropout)

        

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        

        self.fc2 = nn.Linear(hidden_dim, 1)

        

        self.dropout = nn.Dropout(dropout)

               

    def forward(self, text, text_lengths):

        

        # text = [sent len, batch size]

        

        embedded = self.embedding(text)

        

        # embedded = [sent len, batch size, emb dim]

        

        #pack sequence

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,enforce_sorted = False)    

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        

        # unpack sequence

        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        

        # output = [sent len, batch size, hid dim * num directions]

        # output over padding tokens are zero tensors

        

        # hidden = [num layers * num directions, batch size, hid dim]

        # cell = [num layers * num directions, batch size, hid dim]

        

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers

        # and apply dropout

        

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        output = self.fc1(hidden)

        output = self.dropout(self.fc2(output))

                

        #hidden = [batch size, hid dim * num directions]

            

        return output
#creating instance of our LSTM_net class



model = LSTM_net(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
print(model)



#No. of trianable parameters

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    

print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)
TEXT.pad_token

TEXT.vocab.stoi[TEXT.pad_token]
#  to initiaise padded to zeros

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
def binary_accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division 

    acc = correct.sum() / len(correct)

    return acc
# training function 

def train(model, iterator):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        text, text_lengths = batch.text

        #print('text = ', text.shape, 'text_length = ',text_lengths.shape)

        #print('length of text_lengths : ', len(text_lengths) )

        optimizer.zero_grad()

        predictions = model(text, text_lengths).squeeze(1)

        temp = model(text, text_lengths)

        #print('squeeze : ', predictions)

        #print('unsqueezed : ', temp)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)



        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        



    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator):

    

    epoch_loss = 0

    epoch_acc = 0

    model.eval()

    

    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            

            #compute loss and accuracy

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            

            #keep track of loss and accuracy

            epoch_loss += loss.item()

            epoch_acc += acc.item()

            

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

model.to(device) #CNN to GPU





# Loss and optimizer

criterion = nn.BCEWithLogitsLoss()



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





t = time.time()

loss=[]

acc=[]

val_acc=[]

best_valid_loss = float('inf')



for epoch in range(3):

    

    train_loss, train_acc = train(model, train_iterator)

    valid_loss, valid_acc = evaluate(model, valid_iterator)

    

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

    

    loss.append(train_loss)

    acc.append(train_acc)

    val_acc.append(valid_acc)

    

    #save the best model

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'best_model.pt')

        

print(f'time:{time.time()-t:.3f}')
#inference 

import spacy

nlp = spacy.load('en')



def predict(model, sentence):

    #print(sentence)

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence

    length = [len(indexed)]*len(indexed)                                    #compute no. of words

    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor

    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words

    length_tensor = torch.LongTensor(length).to(device)                  #convert to tensor

    

    #print(tensor)

    #print(length_tensor)

    #print(len(length_tensor))

    

    #print(tensor.shape)

    #print(length_tensor.shape)

    prediction = model(tensor, length_tensor).squeeze(1)       #prediction 

    

    rounded_preds = torch.round(torch.sigmoid(prediction))

    predict_class = rounded_preds.tolist()[0]

    return predict_class        
predict(model, 'test message')
for batch in train_iterator:

    text, text_lengths = batch.text

    print(text)

    print(text_lengths)

    print('text = ', text.shape, 'text_length = ',text_lengths.shape)

    print('length of text_lengths : ', len(text_lengths) )

    predictions = model(text, text_lengths).squeeze(1)

    break
test.text[0]

type(test.text[0])
predicts = []

cnt = 0

        

for batch in test_iter:

    text, text_lengths = batch.text

    predictions = model(text, text_lengths).squeeze(1)

    rounded_preds = torch.round(torch.sigmoid(predictions))

    predict_class = rounded_preds.tolist()[0]

    predicts.append(int(predict_class))

    cnt += 1

    

print(cnt)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

submission['target'] = predicts

submission
submission.describe()
submission.to_csv('./submission.csv', index=False)