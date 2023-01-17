import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchtext.data import Field,BucketIterator,TabularDataset
import torchtext
import spacy
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
train = data[:100000]
validation = data[100000:150000]
train.head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",'i\'m':'i am', "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled'}
def clean_contractions(text, mapping):
    text = text.lower()
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else mapping[t.lower()] if t.lower() in mapping else t for t in text.split(" ")])
    return text
def remove_newlines(sent):
    sent = re.sub(r'\s+', " ", sent )
    return sent
train['question1'] = train['question1'].apply(lambda x: clean_contractions(str(x),contraction_mapping))
train['question2'] = train['question2'].apply(lambda x: clean_contractions(str(x),contraction_mapping))
validation['question1'] = validation['question1'].apply(lambda x: clean_contractions(str(x),contraction_mapping))
validation['question2'] = validation['question2'].apply(lambda x: clean_contractions(str(x),contraction_mapping))

train['question1'] = train['question1'].apply(lambda x: remove_newlines(str(x)))
train['question2'] = train['question2'].apply(lambda x: remove_newlines(str(x)))
validation['question1'] = validation['question1'].apply(lambda x: remove_newlines(str(x)))
validation['question2'] = validation['question2'].apply(lambda x: remove_newlines(str(x)))
test = validation.loc[140000:149999]
def view_data(index):
    print(train['question1'][index])
    print(train['question2'][index])
    print("Similarity: ",train['is_duplicate'][index])
    
view_data(77)
print("Training Samples: ",len(train))
print("Validation Samples: ",len(validation))
print("Testing Samples: ",len(test))
# max_length = 0
# for index,row in data.iterrows():
#     if index >= 150000:
#         break
#     max_length = max(len(str(row['question1']).split()),max_length)
#     max_length = max(len(str(row['question2']).split()),max_length)
duplicate_vals = train['is_duplicate']
validation.drop(['id','qid1','qid2'],inplace=True,axis = 1)
train.drop(['id','qid1','qid2'],inplace=True,axis = 1)
test.drop(['id','qid1','qid2'],inplace=True,axis = 1)
train.to_csv('train.csv',index = False)
validation.to_csv('validation.csv',index = False)
test.to_csv('test.csv',index = False)
from torchtext import data
tokenizer = lambda s: s.split()

text1 = data.Field(tokenize=tokenizer,
                  batch_first=True,
                  include_lengths=True,
                  )

text2 = data.Field(tokenize=tokenizer,
                  batch_first=True,
                  include_lengths=True,
                  )

label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
fields = [('question1',text1),('question2',text2),('is_duplicate',label)]
train_data, valid_data,test_data = data.TabularDataset.splits(
    path='/kaggle/working/',
    train='train.csv',
    validation = 'validation.csv',
    test='test.csv',
    format='csv',
    fields=fields,
    skip_header=True
)
print(vars(test_data[0]))
text1.build_vocab(train_data,valid_data)
text2.build_vocab(train_data,valid_data)
print(len(text1.vocab))
print(len(text2.vocab))
label.build_vocab(train_data,valid_data)
# !rm -rf /kaggle/working/crawl-300d-2M.vec.pt
!cp ../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt /kaggle/working 
from torchtext import vocab

embeddings = vocab.Vectors('glove.6B.100d.txt','/kaggle/working')
text1.build_vocab(train_data,test_data,vectors = embeddings)
text2.build_vocab(train_data,test_data,vectors = embeddings)
label.build_vocab()
train_itr,valid_itr,test_itr = data.BucketIterator.splits((train_data,valid_data,test_data),
                                                          batch_size = 32,
                                                          sort_key = lambda x: len(x.question1),
                                                          sort_within_batch = True,
                                                          device = device
                                                          )
def create_embedding_matrix(field,embeddings):  
    embedding_matrix = np.random.rand(len(field.vocab.stoi),100)
    for string,index in field.vocab.stoi.items():
        if not  all(x == 0 for x in embeddings[string].tolist()):
            embedding_matrix[index] = embeddings[string] 
    return embedding_matrix
embedding1 = create_embedding_matrix(text1,embeddings)
embedding2 = create_embedding_matrix(text2,embeddings)
print(embedding1.shape)
print(embedding2.shape)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,pad_index_1,pad_index_2,batch_size,vocab_size_1,vocab_size_2,embedding_matrix1,embedding_matrix2,embedding_dimensions,hidden_size,bidirectional,first_linear_dims,second_linear_dims,third_linear_dims,num_layers):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_dimensions = embedding_dimensions
        self.embedding1 = nn.Embedding(vocab_size_1,embedding_dimensions,padding_idx = pad_index_1)
        self.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix1,dtype=torch.float32))
        self.embedding1.weight.requires_grad = False
        self.embedding2 = nn.Embedding(vocab_size_2,embedding_dimensions,padding_idx = pad_index_2)
        self.embedding2.weight = nn.Parameter(torch.tensor(embedding_matrix2,dtype=torch.float32))
        self.embedding2.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dimensions,hidden_size,batch_first = True,bidirectional = self.bidirectional,num_layers = self.num_layers)
        self.lstm2 = nn.LSTM(embedding_dimensions,hidden_size,batch_first = True,bidirectional = self.bidirectional,num_layers = self.num_layers)
        self.num_directions  = (1 if self.bidirectional == False else 2)
        self.linear1 = nn.Linear(self.hidden_size*self.num_directions,first_linear_dims) #for 1st lstm 
        self.linear2 = nn.Linear(self.hidden_size*self.num_directions,first_linear_dims) #for 2nd lstm
        self.concat_layer = nn.Linear(2*first_linear_dims,second_linear_dims) #for concatenating both the outputs
        self.linear3 = nn.Linear(second_linear_dims,third_linear_dims)#for prediction
        self.final_layer = nn.Linear(third_linear_dims,1)
        
    
    def forward(self,text1,text2,question1_length,question2_length):
        embedded_outputs_1 = self.embedding1(text1)
        embedded_outputs_2 = self.embedding2(text2)
        h_n_1,c_n_1,h_n_2,c_n_2 = self.init_hidden()
        h_n_1 = h_n_1.view(self.num_directions,self.batch_size,self.hidden_size)
        c_n_1 = c_n_1.view(self.num_directions,self.batch_size,self.hidden_size)
        h_n_2 = h_n_2.view(self.num_directions,self.batch_size,self.hidden_size)
        c_n_2 = c_n_2.view(self.num_directions,self.batch_size,self.hidden_size)
        
        
        output_1,(h_n_1,c_n_1) = self.lstm1(embedded_outputs_1,(h_n_1,c_n_1))
        output_2,(h_n_2,c_n_2) = self.lstm2(embedded_outputs_2,(h_n_2,c_n_2))
        
        output_1 = output_1[:,-1,:]
        output_2 = output_2[:,-1,:]
        
        output_1 = self.linear1(output_1.view(self.batch_size,1,self.num_directions*self.hidden_size))
        output_2 = self.linear2(output_2.view(self.batch_size,1,self.num_directions*self.hidden_size))
        
        concatenated_outputs = torch.cat((output_1,output_2),dim = 2)
        concatenated_outputs = concatenated_outputs.view(concatenated_outputs.shape[0],concatenated_outputs.shape[2]) 
        concatenated_logits = self.concat_layer(concatenated_outputs)
        linear3_outputs = self.linear3(concatenated_logits)
        predictions = self.final_layer(linear3_outputs)
        return predictions,h_n_1,c_n_1,h_n_2,c_n_2
    
    def init_hidden(self):
        multiplier = 1
        if self.bidirectional:
            multiplier = 2
        return torch.zeros(self.batch_size,self.num_layers*multiplier,self.hidden_size,dtype = torch.float32,device = device),torch.zeros(self.batch_size,self.num_layers*multiplier,self.hidden_size,dtype = torch.float32,device = device),torch.zeros(self.batch_size,self.num_layers*multiplier,self.hidden_size,dtype = torch.float32,device = device),torch.zeros(self.batch_size,self.num_layers*multiplier,self.hidden_size,dtype = torch.float32,device = device)
        
            
model = Model(pad_index_1 = text1.vocab.stoi[text1.pad_token] ,
              pad_index_2 = text2.vocab.stoi[text2.pad_token] , 
              batch_size = 32,
              vocab_size_1 = len(text1.vocab),
              vocab_size_2 = len(text2.vocab),
              embedding_matrix1 = embedding1,
              embedding_matrix2 = embedding2,
              embedding_dimensions = 100,
              hidden_size = 128,
              bidirectional = True ,
              first_linear_dims = 64,
              second_linear_dims = 32,
              third_linear_dims = 16,
              num_layers = 1
              )

model = model.to(device = device)
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) 
epochs = 30
def return_accuracy(logits,label):
    sigmoid = nn.Sigmoid()(logits)
    predictions = torch.round(sigmoid)
    predictions = predictions.view(32)
    return (predictions == label).sum().float()/float(label.size(0))
def train(epochs,criterion,optimizer,model,train_iterator,valid_iterator):
    
    for epoch in range(epochs):
        print("Epoch {} out of {}".format(epoch,epochs))
        
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        
        epoch_valid_loss = 0
        epoch_valid_accuracy = 0
        
        for batch in train_iterator:
            model.train()
            optimizer.zero_grad()
            question1 = batch.question1[0]
            question2 = batch.question2[0]
            label = batch.is_duplicate
            
            question1_length = batch.question1[1]
            question2_length = batch.question2[1]
            
            question1.to(device)
            question2.to(device)
            label.to(device)

            label = torch.tensor(label,dtype= torch.float32,device = device)
            
            predictions,h_n_1,c_n_1,h_n_2,c_n_2 = model(question1,question2,question1_length,question2_length)
            loss = criterion(predictions,label.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()/len(batch)
            batch_accuracy = return_accuracy(predictions,label)
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += batch_accuracy.item()
            
        print("Epoch Train Accuracy: ",epoch_train_accuracy/len(train_iterator))
        print("Epoch Train Loss: ",epoch_train_loss/len(train_iterator))
    
        for batch_v in valid_iterator:
            model.eval()
            question1_v = batch.question1[0]
            question2_v = batch.question2[0]
            label_v = batch.is_duplicate
            
            question1_length_v = batch.question1[1]
            question2_length_v = batch.question2[1]
            
            question1_v.to(device)
            question2_v.to(device)
            label_v.to(device)

            label_v = torch.tensor(label_v,dtype= torch.float32,device = device)
            
            predictions_v,h_n_1_v,c_n_1_v,h_n_2_v,c_n_2_v = model(question1_v,question2_v,question1_length_v,question2_length_v)
            loss_v = criterion(predictions_v,label_v.unsqueeze(1))
            
            batch_loss_v = loss_v.item()/len(batch_v)
            batch_accuracy_v = return_accuracy(predictions_v,label_v)
            
            epoch_valid_loss += loss_v.item()
            epoch_valid_accuracy += batch_accuracy_v.item()
            
        print("Epoch valid Accuracy: ",epoch_valid_accuracy/len(valid_iterator))
        print("Epoch valid Loss: ",epoch_valid_loss/len(valid_iterator))
        print("--"*60)    
train(epochs,criterion,optimizer,model,train_itr,valid_itr)
def predict(test_iterator,model):
    epoch_test_loss = 0
    epoch_test_accuracy = 0
    for batch in test_iterator:
        model.eval()
        question1 = batch.question1[0]
        question2 = batch.question2[0]
        label = batch.is_duplicate
        
        question1_length = batch.question1[1]
        question2_length = batch.question2[1]

        question1.to(device)
        question2.to(device)
        label.to(device)

        label = torch.tensor(label,dtype= torch.float32,device = device)

        predictions,h_n_1,c_n_1,h_n_2,c_n_2 = model(question1,question2,question1_length,question2_length)
        
        batch_accuracy = return_accuracy(predictions,label)    
        epoch_test_accuracy += batch_accuracy.item()
            
    print("Epoch Test Accuracy: ",epoch_test_accuracy/len(test_iterator))
