import os

import torch

from torchtext import data
SEED = 1234

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
os.listdir("../input/imbd-sentiment-classification-dataset/data")
TEXT = data.Field()

LABEL = data.LabelField(dtype=torch.float)



fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}



train_data, test_data = data.TabularDataset.splits(

    path = '../input/imbd-sentiment-classification-dataset/data',

    train = 'train.json',

    test = 'test.json',

    format = 'json',

    fields = fields

)
import random

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
idx = 4

print("text :"," ".join(train_data[idx].__dict__['text']) )

print("\nlabel :",train_data[idx].__dict__['label'] )
MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)
print(len(TEXT.vocab))

print(len(LABEL.vocab))
TEXT.vocab.stoi
LABEL.vocab.stoi
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device : {device}")

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data,valid_data,test_data),

    batch_size = BATCH_SIZE,

    sort = False,

    device = device, # last comma is optional

)
import torch.nn as nn



class RNN(nn.Module):



    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()



        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim,output_dim)



    def forward(self, text):

        embedded = self.embedding(text)

        output, hidden = self.rnn(embedded)



        assert torch.equal(output[-1,:,:], hidden.squeeze(0))



        return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1



model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
def count_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f"No. of trainable parameters : {count_params(model):,}")
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)

criterion = criterion.to(device)
def bin_acc(preds, y):

    rounded = torch.round(torch.sigmoid(preds))

    correct = (rounded==y).float()

    return sum(correct)/len(correct)
def train(model, iterator, optimizer, criterion):



    epoch_loss = 0

    epoch_acc = 0

    

    model.train()    



    for batch in iterator:

        optimizer.zero_grad()



        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = bin_acc(predictions, batch.label)



        loss.backward()

        optimizer.step()



        epoch_loss += loss

        epoch_acc += acc



    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def evaluate(model, iterator, criterion):



    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()    

    with torch.no_grad():

        for batch in iterator:



            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = bin_acc(predictions, batch.label)



            epoch_loss += loss

            epoch_acc += acc



    return epoch_loss/len(iterator), epoch_acc/len(iterator)
import time



def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
N_EPOCHS = 5



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):



    start_time = time.time()

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    

    end_time = time.time()



    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'simple_rnn.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')