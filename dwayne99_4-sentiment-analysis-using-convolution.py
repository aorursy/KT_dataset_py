import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# NECESSARY DEPENDANCIES

import torch

from torch import nn, optim

from torchtext import data, datasets

import torch.nn.functional as F



import random

import numpy as np



SEED = 1234



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize = 'spacy', batch_first = True)

LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT,LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
# builds a vocab of most common 25k words

MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(

    train_data,

    max_size = MAX_VOCAB_SIZE,

    vectors = 'glove.6B.100d',

    unk_init = torch.Tensor.normal_

)



LABEL.build_vocab(train_data)
BATCH_SIZE = 64



train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size = BATCH_SIZE,

    device = device

)
class Net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.conv_0 = nn.Conv2d(

            in_channels = 1, out_channels = n_filters,

            kernel_size = (filter_sizes[0],embedding_dim))

        

        self.conv_1 = nn.Conv2d(

            in_channels = 1, out_channels = n_filters,

            kernel_size = (filter_sizes[1], embedding_dim)

        )

        

        self.conv_2 = nn.Conv2d(

            in_channels = 1, out_channels = n_filters,

            kernel_size = (filter_sizes[2], embedding_dim)

        )

        

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

        

    def forward(self, text):

        

        # text = [batch_size, sent_len]

        embedded = self.embedding(text)

        # embedded = [batch_size, sent_len, embedding_dim]

        

        # we do this to create an additional dim for the channels

        embedded = embedded.unsqueeze(1)

        # embedded = [batch_size, 1, sent_len, embedding_dim]

        

        conv0 = F.relu(self.conv_0(embedded).squeeze(3))

        conv1 = F.relu(self.conv_1(embedded).squeeze(3))

        conv2 = F.relu(self.conv_2(embedded).squeeze(3))

        

        # conv = [batch_size, n_filters, sent_len - filter_size + 1]

        

        pool0 = F.max_pool1d(conv0, conv0.shape[2]).squeeze(2)

        pool1 = F.max_pool1d(conv1, conv1.shape[2]).squeeze(2)

        pool2 = F.max_pool1d(conv2, conv2.shape[2]).squeeze(2)

        

        # pool_n = [batch_size, n_filters]

        

        cat = self.dropout(torch.cat((pool0,pool1, pool2), dim=1))

        

        # cat = [batch_size, n_filters * len(filter_sizes)]

        

        return self.fc(cat)
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

N_FILTERS = 100

FILTER_SIZES = [3,4,5]

OUTPUT_DIM = 1

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = Net(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
# Select Optimizer

optimizer = optim.Adam(model.parameters())



# Select the loss function

criterion = nn.BCEWithLogitsLoss()



model = model.to(device)

criterion = criterion.to(device)
def check_correct(preds,labels):

    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    return (rounded_preds == labels).float().sum().item()  
def train(model,iterator,optimizer,criterion):

    

    epoch_loss, correct, total = 0,0,0

    

    model.train()

    

    for batch in tqdm(iterator):

        

        total += len(batch)

        

        optimizer.zero_grad()

        

        text = batch.text

        

        predictions = model(text).squeeze(1)

        

        loss = criterion(predictions, batch.label)

        

        correct += check_correct(predictions, batch.label)

        

        loss.backward()

        

        optimizer.step()

        

        epoch_loss += loss.item()



    return epoch_loss / len(iterator), correct / total
def evaluate(model, iterator, criterion):

    

    epoch_loss, correct, total = 0,0,0

    

    model.eval()

    

    with torch.no_grad():

    

        for batch in tqdm(iterator):

            

            total += len(batch)



            text = batch.text

            

            predictions = model(text).squeeze(1)

            

            loss = criterion(predictions, batch.label)

            

            correct += check_correct(predictions, batch.label)



            epoch_loss += loss.item()

        

    return epoch_loss / len(iterator), correct / total
from tqdm import tqdm



N_EPOCHS = 5



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, val_iterator, criterion)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'model.pt')

    

    print(f'Epoch: {epoch+1:02}')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
model.load_state_dict(torch.load('model.pt'))



test_loss, test_acc = evaluate(model, test_iterator, criterion)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
import spacy 

spacy = spacy.load('en')



def predict_sentiment(model, sentence, min_length = 5):

    

    # set model to evaluation mode

    model.eval()

    

    # tokenize the text

    tokenized = [token.text for token in spacy.tokenizer(sentence)]

    

    # check if sentence less than min_length

    if len(tokenized) < min_length:

        tokenized += ['<pad>'] * (min_length - len(tokenized))

        

    # index the tokens

    indexed = [ TEXT.vocab.stoi[token] for token in tokenized]

    

    # convert input to tensor

    tensor = torch.LongTensor(indexed).to(device)

    

    # unsqueeze to create a tensor of batch_size = 1

    tensor = tensor.unsqueeze(0)

    

    # output

    prediction = torch.sigmoid(model(tensor))

    

    #class

    label = torch.round(prediction).item()

    

    if label:

        print(f'Positive review with a probability of {prediction.item():.3f}')

    else:

        print(f'Negative review with a probability of {1-prediction.item():.3f}')
predict_sentiment(model, "This film is terrible")
predict_sentiment(model, "This film is great")