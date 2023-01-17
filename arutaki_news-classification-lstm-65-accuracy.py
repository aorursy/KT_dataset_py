import torch

from torchtext import data
TEXT = data.Field(tokenize = 'spacy', lower = True)

LABEL = data.LabelField()
news = data.TabularDataset(

    path='../input/News_Category_Dataset_v2.json', format='json',

    fields={'headline': ('headline', TEXT),

            'short_description' : ('desc', TEXT),

             'category': ('category', LABEL)})
import random

SEED = 1234



trn, vld, tst = news.split(split_ratio=[0.7, 0.2, 0.1], random_state = random.seed(SEED))
vars(trn[0])
TEXT.build_vocab(trn, 

                 vectors = "glove.6B.100d", 

                 unk_init = torch.Tensor.normal_)



LABEL.build_vocab(trn)
print(len(TEXT.vocab))

print(len(LABEL.vocab))
print(LABEL.vocab.stoi)
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(

    (trn, vld, tst), 

    batch_size = BATCH_SIZE, 

    device = device,

    sort_key= lambda x: len(x.headline), 

    sort_within_batch= False

    )
import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        

        super().__init__()

                

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        

        self.lstm_head = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout)

        

        self.lstm_desc = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout)

        

        self.fc_head = nn.Linear(hidden_dim * 2, 100)

        

        self.fc_desc = nn.Linear(hidden_dim * 2, 100)

        

        self.fc_total = nn.Linear(200, output_dim)

        

        self.dropout = nn.Dropout(dropout)

                

    def forward(self, headline, description):

                        

        embedded_head = self.dropout(self.embedding(headline))

        

        embedded_desc = self.dropout(self.embedding(description))

                                    

        output_head, (hidden_head, cell_head) = self.lstm_head(embedded_head)

        

        output_desc, (hidden_desc, cell_desc) = self.lstm_desc(embedded_desc)

        

        hidden_head = self.dropout(torch.cat((hidden_head[-2, :, :], hidden_head[-1, :, :]), dim = 1))

        

        hidden_desc = self.dropout(torch.cat((hidden_desc[-2, :, :], hidden_desc[-1, :, :]), dim = 1))

        

        full_head = self.fc_head(hidden_head)

        

        full_desc = self.fc_desc(hidden_desc)

        

        hidden_total = torch.cat((full_head, full_desc), 1)

        

        return self.fc_total(hidden_total)
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = len(LABEL.vocab)

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.2



model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
import torch.optim as optim



optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()



model = model.to(device)

criterion = criterion.to(device)
def categorical_accuracy(preds, y):

    max_preds = preds.argmax(dim = 1, keepdim = True)

    correct = max_preds.squeeze(1).eq(y)

    return correct.sum() / torch.FloatTensor([y.shape[0]])
def train(model, iterator, optimizer, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        

        optimizer.zero_grad()

                        

        predictions = model(batch.headline, batch.desc).squeeze(1)

        

        loss = criterion(predictions, batch.category)

        

        acc = categorical_accuracy(predictions, batch.category)

        

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

            

            predictions = model(batch.headline, batch.desc).squeeze(1)

            

            loss = criterion(predictions, batch.category)

            

            acc = categorical_accuracy(predictions, batch.category)



            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

        torch.save(model.state_dict(), 'news_classification_model.pt')

    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

test_loss, test_acc = evaluate(model, test_iterator, criterion)



print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
import spacy

nlp = spacy.load('en')



def predict_category(model, head, desc):

    model.eval()

    head = head.lower()

    desc = desc.lower()

    tokenized_head = [tok.text for tok in nlp.tokenizer(head)]

    tokenized_desc = [tok.text for tok in nlp.tokenizer(desc)]

    indexed_head = [TEXT.vocab.stoi[t] for t in tokenized_head]

    indexed_desc = [TEXT.vocab.stoi[t] for t in tokenized_desc]

    tensor_head = torch.LongTensor(indexed_head).to(device)

    tensor_desc = torch.LongTensor(indexed_desc).to(device)

    tensor_head = tensor_head.unsqueeze(1)

    tensor_desc = tensor_desc.unsqueeze(1)

    prediction = model(tensor_head, tensor_desc)

    max_pred = prediction.argmax(dim=1)

    return max_pred.item()
pred = predict_category(model, "Trump’s Art Of Distraction", "The conversation surrounding Trump’s latest racist rants has provoked us to revisit author Toni Morrison’s 1975 keynote address at Portland State University on the true purpose of racism..")

print(f'Predicted category is: {pred} = {LABEL.vocab.itos[pred]}')
pred = predict_category(model, "Indiana Cop Apologizes After Accusing McDonald’s Worker Of Eating His Sandwich", "The Marion County sheriff’s deputy forgot he had taken a bite out of his McChicken earlier that day, authorities said.")

print(f'Predicted category is: {pred} = {LABEL.vocab.itos[pred]}')
pred = predict_category(model, "Kyle ‘Bugha’ Giersdorf, 16, Wins Fortnite World Cup And Takes Home $ 3 Million Prize", "Fortnite has nearly 250 million registered players and raked in an estimated $2.4 billion last year.")

print(f'Predicted category is: {pred} = {LABEL.vocab.itos[pred]}')