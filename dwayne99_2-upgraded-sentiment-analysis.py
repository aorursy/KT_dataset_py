import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# NECESSARY DEPENDANCIES

import torch

from torchtext import data, datasets

import random

from torch import nn



from tqdm import tqdm



SEED = 1234

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = data.Field(tokenize = 'spacy', include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)



# Further splitting the training data into train and validation sets

train_data, val_data = train_data.split(random_state = random.seed(SEED))



print(f'Training size = {len(train_data)}')

print(f'Testing size = {len(test_data)}')

print(f'Validation size = {len(val_data)}')
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

    (train_data,val_data,test_data),

    batch_size = BATCH_SIZE,

    sort_within_batch = True, 

    device = device

)
class LSTM(nn.Module):

    

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        

        self.lstm = nn.LSTM(

            embedding_dim, hidden_dim,

            num_layers = n_layers,

            bidirectional = bidirectional,

            dropout = dropout)

        

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text, text_lengths):

        

        #text = [sent len, batch size]

        

        embedded = self.dropout(self.embedding(text))

        

        #embedded = [sent len, batch size, emb dim]

        

        #pack sequence

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        

        #unpack sequence

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)



        #output = [sent len, batch size, hid dim * num directions]

        #output over padding tokens are zero tensors

        

        #hidden = [num layers * num directions, batch size, hid dim]

        #cell = [num layers * num directions, batch size, hid dim]

        

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers

        #and apply dropout

        

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

                

        #hidden = [batch size, hid dim * num directions]

            

        return self.fc(hidden)
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = LSTM(INPUT_DIM, 

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
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
from torch import optim



optimizer = optim.Adam(model.parameters())
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

        

        text, text_lengths = batch.text

        

        predictions = model(text, text_lengths).squeeze(1)

        

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



            text, text_lengths = batch.text

            

            predictions = model(text, text_lengths).squeeze(1)

            

            loss = criterion(predictions, batch.label)

            

            correct += check_correct(predictions, batch.label)



            epoch_loss += loss.item()

        

    return epoch_loss / len(iterator), correct / total
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



def predict_sentiment(model, sentence):

    

    # set model to evaluation mode

    model.eval()

    

    # tokenize the sentence

    tokenized = [token.text for token in spacy.tokenizer(sentence)]

    

    # index the tokens

    indexed = [TEXT.vocab.stoi[token] for token in tokenized]

    

    # length of sentence

    length = [len(indexed)]

    

    # convert the input and length to tensors

    tensor = torch.LongTensor(indexed).to(device)

    length_tensor = torch.LongTensor(length)

    

    # create a batch of size 1 to match model dimentions

    tensor = tensor.unsqueeze(1)

    

    prediction = torch.sigmoid(model(tensor, length_tensor))

    

    value = torch.round(prediction).item()

    

    if value:

        print(f'The review is postive with a probability of {prediction.item()}')

    else:

        print(f'The review is negative with a probability of {1 - prediction.item()}')
predict_sentiment(model, "This film is terrible")


predict_sentiment(model, "This film is great")