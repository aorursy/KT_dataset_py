import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Function to obtain Bi-grams

def generate_bigrams(tokens):

    '''

    The generate_bigrams function takes a sentence that has already been tokenized,

    calculates the bi-grams and appends them to the end of the tokenized list.

    '''

    x = list(set(tokens))

    bigrams = [tokens[i] + ' ' + tokens[i + 1] for i in range(len(tokens) - 1)]

    return x + bigrams

    

# test the function

generate_bigrams(['This', 'film', 'is', 'terrible'])
import torch

from torchtext import data, datasets

SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams)

LABEL = data.LabelField(dtype = torch.float)
import random



train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)



train_data, valid_data = train_data.split(random_state = random.seed(SEED))
# Build the Vocab



MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(

    train_data,

    max_size = MAX_VOCAB_SIZE,

    vectors = 'glove.6B.100d',

    unk_init = torch.Tensor.normal_

)



LABEL.build_vocab(train_data)
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (train_data, valid_data, test_data),

    batch_size = BATCH_SIZE,

    device = device

)
from torch import nn

from torch.nn import functional as F



class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.fc = nn.Linear(embedding_dim, output_dim)

        

    def forward(self, text):

        

        # text = [sent_len, batch_size]

        embedded = self.embedding(text)

        

        # embedded = [sent_len, batch_size, emb_dim]

        embedded = embedded.permute(1,0,2)

        

        #embedded = [batch_size, sent_len, emb_dim]

        

        pooled = F.avg_pool2d(embedded, (embedded.shape[1],1)).squeeze(1)

        

        #pooled = [batch_size, emb_dim]

        return self.fc(pooled)
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

OUTPUT_DIM = 1

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]



model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#optimizer

import torch.optim as optim



optimizer = optim.Adam(model.parameters())



# loss function

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

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'model.pt')

    

    print(f'Epoch: {epoch+1:02}')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
# Test Set

model.load_state_dict(torch.load('model.pt'))



test_loss, test_acc = evaluate(model, test_iterator, criterion)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
import spacy

spacy = spacy.load('en')



def predict_sentiment(model,sentence):

    

    model.eval()

    tokenized = generate_bigrams([token.text for token in spacy.tokenizer(sentence)])

    indexed = [TEXT.vocab.stoi[token] for token in tokenized]

    

    tensor = torch.LongTensor(indexed).to(device)

    tensor = tensor.unsqueeze(1)

    prediction = torch.sigmoid(model(tensor))

    value = torch.round(prediction).item()

    

    if value:

        print(f'The review is postive with a probability of {prediction.item()}')

    else:

        print(f'The review is negative with a probability of {1 - prediction.item()}')
predict_sentiment(model, "This film is terrible")
predict_sentiment(model, "This film is great")