import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch 

from torchtext import data, datasets

from torch import nn, optim

import torch.nn.functional as F



import random

import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
len(tokenizer.vocab)
tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')



print(tokens)
indexes = tokenizer.convert_tokens_to_ids(tokens)



print(indexes)
init_token = tokenizer.cls_token



# end of sentence token

eos_token = tokenizer.sep_token



# pad token

pad_token = tokenizer.pad_token



# unknown token

unk_token = tokenizer.unk_token



print(init_token, eos_token, pad_token, unk_token)
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)

eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)

pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)

unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)



print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
init_token_idx = tokenizer.cls_token_id

eos_token_idx = tokenizer.sep_token_id

pad_token_idx = tokenizer.pad_token_id

unk_token_idx = tokenizer.unk_token_id



print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

print(max_input_length)
def tokenize_and_cut(sentence):

    

    tokens = tokenizer.tokenize(sentence)

    tokens = tokens[:max_input_length - 2]

    return tokens
from torchtext import data



TEXT = data.Field(batch_first = True,

                  use_vocab = False,

                  tokenize = tokenize_and_cut,

                  preprocessing = tokenizer.convert_tokens_to_ids,

                  init_token = init_token_idx,

                  eos_token = eos_token_idx,

                  pad_token = pad_token_idx,

                  unk_token = unk_token_idx)



LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)



train_data, valid_data = train_data.split(random_state = random.seed(SEED))



print(f"Number of training examples: {len(train_data)}")

print(f"Number of validation examples: {len(valid_data)}")

print(f"Number of testing examples: {len(test_data)}")
print(vars(train_data.examples[6]))
tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])



print(tokens)
LABEL.build_vocab(train_data)



print(LABEL.vocab.stoi)
BATCH_SIZE = 128



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size = BATCH_SIZE,

    device = device

)
from transformers import BertTokenizer, BertModel



bert = BertModel.from_pretrained('bert-base-uncased')
class BertGRU(nn.Module):

    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        

        self.bert = bert

        

        embedding_dim = bert.config.to_dict()['hidden_size']

        

        self.rnn = nn.GRU(embedding_dim, hidden_dim,

                         num_layers=n_layers,bidirectional=bidirectional,

                         batch_first = True, 

                         dropout = 0 if n_layers < 2 else dropout)

        

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text):

        

        # text = [batch_size, sent_len]

        with torch.no_grad():

            embedded = self.bert(text)[0]

            

        # embedded = [batch_size, sent_len, embedding_dim]

        

        _, hidden = self.rnn(embedded)

        

        # hidden = [n_layers * n_directions, batch_size, embedding_dim]

        

        if self.rnn.bidirectional:

            hidden = self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))

            

        else:

            hidden = self.dropout(hidden[-1,:,:])

            

        #hidden = [batch_size, hidden_dim]

        

        output = self.out(hidden)

        

        #output = [batch_size, out_dim]

        

        return output
HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.25



model = BertGRU(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL,DROPOUT)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
for name, param in model.named_parameters():                

    if name.startswith('bert'):

        param.requires_grad = False
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
for name, param in model.named_parameters():                

    if param.requires_grad:

        print(name)
optimizer = optim.Adam(model.parameters())



criterion = nn.BCEWithLogitsLoss()
# place the model on GPU if available

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
def predict_sentiment(model, tokenizer,sentence):

    

    model.eval()

    

    # tokenize the data

    tokens = tokenizer.tokenize(sentence)

    

    # limit the length of the tokens

    tokens = tokens[:max_input_length-2]

    

    # index the tokens

    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]

    

    # convert to tensor

    tensor = torch.LongTensor(indexed).to(device)

    

    # create a batch dimention

    tensor = tensor.unsqueeze(0)

    

    # forward pass

    prediction = torch.sigmoid(model(tensor))

    

    #class

    label = torch.round(prediction).item()

    

    if label:

        print(f'Positive review with a probability of {prediction.item():.3f}')

    else:

        print(f'Negative review with a probability of {1-prediction.item():.3f}')
predict_sentiment(model, tokenizer, "This film is terrible")
predict_sentiment(model, tokenizer, "This film is great")