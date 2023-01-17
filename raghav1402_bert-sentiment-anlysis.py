from transformers import BertModel,AutoTokenizer,BertTokenizer
import torch
import pandas as pd
import transformers
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import datetime
import time
import matplotlib.pyplot as plt
%matplotlib inline
print('Torch version : ', torch.__version__)
print('transformers version : ', transformers.__version__)

torch.cuda.is_available()
pretrained_weights = '/kaggle/input/bert-base-uncased/'
# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
bert_model = BertModel.from_pretrained(pretrained_weights,max_position_embeddings=512)
sentence_list = ["This is a sample", "This is another longer sample text"]
tokens = tokenizer.batch_encode_plus(sentence_list,
    pad_to_max_length=True  # Smallest sentences will be padded to match the longest sentnece
)
encoded_dict = tokenizer.encode_plus("This is a sample", "This is another longer sample text",max_length =15, pad_to_max_length=True)
print(encoded_dict)
'''Iterate through every sentence to get the embedding '''
for i in range(len(sentence_list)):
    input_ids  = torch.tensor([(tokens['input_ids'][i])])
    segments_tensors = torch.tensor([[i+1] * len(input_ids[0])])
    print("input_ids : ",input_ids.size())    
    with torch.no_grad():
        '''the hidden unit / feature number (768 features) for embedding using bert-base-cased '''
        '''Pass the input_tokens to the model to get the embeddings'''
        #Predict hidden states features for each layer
        ''' BERT outputs two tensors:
            One with the generated representation for every token in the input (1, NB_TOKENS, REPRESENTATION_SIZE)
            One with an aggregated representation for the whole input (1, REPRESENTATION_SIZE)'''
        encoded_layers, pooled = bert_model(input_ids,segments_tensors)#[0]
    print("Embedding size :", encoded_layers.size()) 
    print(pooled.size())
data_df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
data_df['sentiment'] = data_df.apply(lambda row: 1 if row['sentiment'] =='positive' else 0, axis=1)
'''randomly picks 5000 records from the dataframe'''
subset_df = data_df.sample(n=3000).reset_index(drop = True)
subset_df.head()
train_sentence_list = list(subset_df['review'][:100])
train_sentiment_list = list(subset_df['sentiment'][:100])

valid_sentence_list = list(subset_df['review'][100: 150])
valid_sentiment_list = list(subset_df['sentiment'][100: 150])
pos_train = len([y for y in train_sentiment_list if y ==1])
pos_valid = len([y for y in valid_sentiment_list if y ==1])
print("Positive sentiment in train : ",pos_train)
print("Positive sentiment in valid : ",pos_valid)
class TextDataset(Dataset):
    def __init__(self, sentence_list, score_list, max_len = 512):
        self.sentence_list = sentence_list
        self.sentiment = score_list
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.max_len = max_len
        print("Total sentences", len(self.sentence_list))
                
    def _get_tokenized_sent_list(self, batch):
        sent_list = []
        sentiment_list = []
        for sent_idx, senti in batch:
            sent_list.append(self.sentence_list[sent_idx.item()][:self.max_len])
            sentiment_list.append(senti.item())      
            
        tokens_list = self.tokenizer.batch_encode_plus(sent_list, 
            pad_to_max_length=True  # Smallest sentences will be padded to match the longest sentnece
        )        
        return torch.tensor(tokens_list['input_ids']), torch.tensor(sentiment_list)
    
    def __len__(self):
        return len(self.sentence_list)    
    def __getitem__(self, idx):
        sentiment_score = torch.tensor(self.sentiment[idx])
        sent_idx = torch.tensor(idx)
        return sent_idx,sentiment_score
train_data_set = TextDataset(train_sentence_list, train_sentiment_list)
valid_data_set= TextDataset(valid_sentence_list, valid_sentiment_list)
batch_size=8
embedding_size = 768
enc_hidd_size = 512
gru_is_bidirectional = True
num_layers =2
num_epochs = 2
dropout = 0.3
train_loader = DataLoader(train_data_set, batch_size=batch_size, \
                          shuffle=True, num_workers=0, drop_last=True, collate_fn=train_data_set._get_tokenized_sent_list)

valid_loader = DataLoader(valid_data_set, batch_size=batch_size, \
                          shuffle=True, num_workers=0, drop_last=True, collate_fn=valid_data_set._get_tokenized_sent_list)
class SentimentAnalysisGRU(nn.Module):
    def __init__(self, p_enc_hidd_size, p_embedding_size, num_layers,dropout, p_is_bidirectional):
        
        super(SentimentAnalysisGRU, self).__init__()
        
        
        self.hidden_size = p_enc_hidd_size
        self.num_layers = num_layers
        self.is_bidirectional = p_is_bidirectional
        ''' Embedding layer'''    
        self.bert_embedding = BertModel.from_pretrained(pretrained_weights,max_position_embeddings=512)
        
        for name, param in self.bert_embedding.named_parameters():                
            param.requires_grad = False
        
        ''' GRU cell'''
        self.gru = nn.GRU(input_size = p_embedding_size, hidden_size = p_enc_hidd_size, num_layers=self.num_layers,\
                          bidirectional = p_is_bidirectional)
        
        self.droupout = nn.Dropout(dropout)
        self.bi_dir = 1
        if self.is_bidirectional:
            self.bi_dir = 2
        '''#output of linear layer is 1 value (pos/neg) '''   
        self.fc = nn.Linear(p_enc_hidd_size * self.bi_dir, out_features=1)
        self.sigmoid = nn.Sigmoid()   
    
    def forward(self, sent_batch):
        #print(batch[0])
        batch_size = sent_batch.size()[0]
#         print(batch_size)
        '''#Embedding size : (batch_size, tokens, 768)'''
        with torch.no_grad():
            embedding, pooled = self.bert_embedding(sent_batch)
#         print("Embedding size : ", embedding.size())
        
        '''# h_0 shape => (num_layers * num_directions, batch, hidden_size)'''
        h_0 = torch.zeros((self.num_layers*self.bi_dir, sent_batch.size()[1], self.hidden_size))
        if self.is_bidirectional:
            h_0 = torch.zeros((self.num_layers * 2, sent_batch.size()[1], self.hidden_size))
        
        '''gru_out : [batch_size,  tokens, gru_hidden*2 (bidirectional)]
        h_n => last stacked hidden state
        h_n => (2, batch_size, gru_hidden) (2 cause, bidirectional and num of layers = 1)'''
        gru_out, h_n = self.gru(embedding, h_0)
#         print(h_n.size())
#         print(gru_out.size())
        
        '''gru_out :  (batch_size*token_len, n_hidden)'''
        
        gru_out = gru_out.contiguous().view(-1, self.hidden_size*self.bi_dir)
        
#         print("After view : ",gru_out.size())
        
        '''out :  (batch_size*seq_length, 1)'''
        fc_out = self.fc(gru_out)
#         print("Linear output size :", out.size())
        
        sig_out = self.sigmoid(fc_out)
#         print('Initial sigmoid output : ',sig_out.size())
        sig_out = sig_out.view(batch_size, -1)
        
#         print("Sigmoid output shape after view : ", sig_out.size())
        '''extract the output of ONLY the LAST output of the LAST element of the sequence'''
        sig_out = sig_out[:, -1]
#         print("Final : ", sig_out.size())
        #print(sig_out)
        return sig_out
model = SentimentAnalysisGRU(enc_hidd_size, embedding_size, num_layers,dropout, gru_is_bidirectional)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
def train(train_loader, model, criterion, optimizer):
    model.train()
    losses = []
    for sentence,label in train_loader:
        output = model(sentence)
#         print("output in train : ", output)
        loss = criterion(output, label.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses)/len(losses)

def valid(valid_loader, model,criterion):
    model.eval()
    losses = []
    for sentence, label in valid_loader:
        with torch.no_grad():
            output = model(sentence)
#             print("output in valid : ",output)
            loss = criterion(output, label.float())
            losses.append(loss.item())
    return sum(losses)/len(losses)
train_loss_list = []
val_loss_list = []
for i in range(0,num_epochs):
    strt = time.time()
    train_loss = train(train_loader, model, criterion, optimizer)
    train_loss_list.append(train_loss)
#     val_loss = valid(valid_loader, model, criterion)
#     val_loss_list.append(val_loss)
    print("train loss : ", train_loss)
#     print("valid loss : ", val_loss)
    end = time.time()
    print("Epoch {} Time taken {} " .format(i+1, round((end-strt),2)))
plt.plot(train_loss_list, '-r')
print("test")
