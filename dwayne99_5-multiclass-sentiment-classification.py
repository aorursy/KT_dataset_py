import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch 

from torchtext import data,datasets

from torch import nn, optim

from torch.nn import functional as F



import random

import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize = 'spacy', batch_first=True)

#When doing a mutli-class problem, PyTorch expects the labels to be numericalized LongTensors.

LABEL = data.LabelField()
train_data, test_data = datasets.TREC.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
#Let's look at one of the examples in the training set.



vars(train_data[-1])
MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(

    train_data,

    max_size = MAX_VOCAB_SIZE,

    vectors = 'glove.6B.100d',

    unk_init = torch.Tensor.normal_

)



LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)
BATCH_SIZE = 64



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size = BATCH_SIZE,

    device = device

)
class CNN(nn.Module):

    

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):

        super().__init__()

        

        # embedding layer

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        # conv layers

        self.convs = nn.ModuleList([

            nn.Conv2d(in_channels = 1,out_channels=n_filters,kernel_size = (fs, embedding_dim)) for fs in filter_sizes

        ])

        

        # fully connected layer

        self.fc = nn.Linear(n_filters*len(filter_sizes),output_dim)

        

        # dropout layer

        self.dropout = nn.Dropout(dropout)

        

    def forward(self,text):

        

        # text = [batch_size, sent_len]

        embed = self.embedding(text)



        # embed =  [batch_size, sent_len, embedding_dim]

        # lets unsqueeze to create a dimention for channel

        embed = embed.unsqueeze(1)

        # embed =  [batch_size, 1, sent_len, embedding_dim]

        

        # pass the embeddings through all the conv layers stored in the modules list

        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]

        # conv_n = [batch_size, n_filters, sent_len - filters + 1]

        

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch_size,n_filters]

        

        cat = self.dropout(torch.cat(pooled, dim=1))

        

        #cat = [batch size, n_filters * len(filter_sizes)]

            

        return self.fc(cat)
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

N_FILTERS = 100

FILTER_SIZES = [2,3,4]

OUTPUT_DIM = len(LABEL.vocab)

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = CNN(INPUT_DIM,EMBEDDING_DIM,N_FILTERS,FILTER_SIZES,OUTPUT_DIM,DROPOUT,PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
# Optimizer

optimizer = optim.Adam(model.parameters())



# Loss Function

criterion = nn.CrossEntropyLoss()



model = model.to(device)

criterion = criterion.to(device)
def check_correct(preds, y):

    max_preds = preds.argmax(dim =1) # get the index of the max probability

    return (max_preds == y).float().sum().item()
def train(model, iterator, criterion, optimizer):

    

    running_loss, total, correct = 0, 0 , 0

    

    # set model to train mode

    model.train()

    

    for batch in iterator:

        

        total += len(batch)

        

        # zero out the accumalated grads

        optimizer.zero_grad()

        

        # Forward Prop

        prediction = model(batch.text)

        

        loss = criterion(prediction, batch.label)

        

        # Back Prop

        loss.backward()

        optimizer.step()

        

        # metrics 

        running_loss += loss.item()

        correct += check_correct(prediction,batch.label)

        

    return running_loss / len(iterator), correct/total        
def evaluate(model, iterator, criterion):

    

    with torch.no_grad():

        

        running_loss, total, correct = 0, 0 , 0

    

        model.eval()

        

        for batch in iterator:



            total += len(batch)



            # zero out the accumalated grads

            optimizer.zero_grad()



            # Forward Prop

            prediction = model(batch.text)

            loss = criterion(prediction, batch.label)



            # metrics 

            running_loss += loss.item()

            correct += check_correct(prediction,batch.label)

        

    return running_loss / len(iterator), correct/total    
from tqdm import tqdm



N_EPOCHS = 5



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    

    train_loss, train_acc = train(model, train_iterator, criterion, optimizer)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    

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



def predict_class(model, sentence, min_length = 4):

    

    with torch.no_grad():

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

        preds = torch.sigmoid(model(tensor))



        #class

        max_preds = preds.argmax(dim = 1).item()



        print(f'Predicted class is:{max_preds} = {LABEL.vocab.itos[max_preds]}')
predict_class(model, "Who is Dwayne Fernandes?")
predict_class(model, "Where is Paris?")
predict_class(model, "How much is four times eight?")
predict_class(model, "What does YOLO stand for?")
predict_class(model, "How do I apply for a license?")