import torch

from torchtext import data

import pandas as pd

import random

import io

import nltk

import pyodbc

nltk.download('stopwords')

SEED = 1234

from nltk.corpus import stopwords

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy', stop_words=stopwords.words('english'))

LABEL = data.LabelField(dtype=torch.float)
#Loading Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')

train.columns
train = train.drop(['Id'], axis=1)

train["review"] = train['review'].str.replace('[^\w\s]','')

train.to_csv('temp.csv', index=False)
fields = [('text', TEXT), ('label', LABEL)]

train_data = data.TabularDataset.splits(

                                        path = '',

                                        format = 'csv',train='temp.csv',

                                        fields = fields,

                                        skip_header = True

)

train_data=train_data[0]
import random

train_data, valid_data = train_data.split(split_ratio=0.7,random_state = random.seed(SEED))
print("Example of training data ")

print(vars(train_data.examples[10]))
print(f'Number of training examples: {len(train_data)}')

print(f'Number of validation examples: {len(valid_data)}')
len(train_data)
TEXT.build_vocab(train_data, max_size=10000)

LABEL.build_vocab(train_data)
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_data, valid_data),

    batch_size=BATCH_SIZE,

    sort_key=lambda x:len(x.text),

    sort_within_batch=False,

    device=device)
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        output, hidden = self.rnn(embedded)

        #output = [sent len, batch size, hid dim]

        #hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
len(TEXT.vocab)
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)

criterion = criterion.to(device)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division

    acc = correct.sum()/len(correct)

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



            #text, text_lengths = batch.text

            

            #text, text_lengths = batch.text

            

            predictions = model(batch.text).squeeze(1)

            

            loss = criterion(predictions, batch.label)

            

            acc = binary_accuracy(predictions, batch.label)



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
N_EPOCHS = 4
for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')



test = pd.read_csv('../input/test.csv')

for sentence in test['review']:

    model.eval()

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    length_tensor = torch.LongTensor(length)

    prediction = torch.sigmoid(model(tensor))

    #print(prediction)

    if prediction>.8:

        #print('1')

        test['sentiment']=1

    else:

        #print('0')

        test['sentiment']=0
len(test)
pdf=test
test.drop(['review'], axis=1).to_csv('submission.csv', index=False)
