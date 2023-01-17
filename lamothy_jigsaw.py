import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
#import keras
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from numba import cuda
import sys
from numpy import save, load
from random import sample
train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
#test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
embedding_matrix = load('/kaggle/input/jigsaw-gru/embedding_matrix.npy')

train = train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify = train.toxic.values, 
                                                  random_state = 42, 
                                                  test_size = 0.1, shuffle = True)
%%time
# using keras tokenizer here
token = text.Tokenizer(num_words = None)
max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen = max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen = max_len)

word_index = token.word_index

#save('/kaggle/working/xtrain_pad.npy', xtrain_pad)
#save('/kaggle/working/xvalid_pad.npy', xvalid_pad)

batch_size = 32
train_data = TensorDataset(torch.LongTensor(xtrain_pad), torch.tensor(ytrain, dtype=torch.int8))
train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
#valid_data = TensorDataset(torch.LongTensor(xvalid_pad), torch.from_numpy(yvalid))
#valid_loader = DataLoader(valid_data, shuffle = False, batch_size = batch_size, drop_last = True)
# %%time
# embeddings_index = {}
# f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
# for line in tqdm(f):
#     values = line.split(' ')
#     word = values[0]
#     coefs = np.asarray([float(val) for val in values[1:]])
#     embeddings_index[word] = coefs
# f.close()

# print('Found %s word vectors.' % len(embeddings_index))

# create an embedding matrix for the words we have in the dataset
# embedding_matrix = np.zeros((len(word_index) + 1, 300))
# for word, i in tqdm(word_index.items()):
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#save('/kaggle/working/embedding_matrix.npy', embedding_matrix)
#embedding_matrix,'jigsaw/embedding_matrix.csv')

embedding_matrix = embedding_matrix.astype('float32')
#del embeddings_index
# local_vars = list(locals().items())
# var_mem = []
# for var, obj in local_vars:
#     tmp = round(sys.getsizeof(obj)/1024**2,2)
#     if tmp > 0:
#         var_mem.append((var, tmp))
    
# var_mem = sorted(var_mem, key = lambda x: x[1], reverse= True)
# var_mem
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#device = torch.device("cpu")
class GRUNet(nn.Module):
    def __init__(self, embd_mat, hidden_size = 300, trainable = False):
        super(GRUNet, self).__init__()
        embd_num, embd_dim = embd_mat.shape
        self.embd = nn.Embedding(embd_num, embd_dim)
        self.embd.load_state_dict({'weight': torch.tensor(embd_mat)})        
        self.embd.weight.requires_grad = trainable
        self.embd_dropout = nn.Dropout2d(0.3)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embd_dim, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, h0):
        embeddings = self.embd_dropout(self.embd(inputs))
        out, h = self.gru(embeddings, h0)
        out = self.sigmoid(self.fc(out[:, -1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(1, batch_size, self.hidden_size).zero_().to(device)
        return hidden
    
    def predict(self, inputs):
        out, h = self.forward(inputs, h0 = self.init_hidden(len(inputs)))
        return out   

def train_model(train_loader, learn_rate = 0.01, EPOCHS = 5):
    
    # Setting common hyperparameters
    #input_dim = next(iter(train_loader))[0].shape[1]
    #output_dim = 1
    
    # Instantiating the models
    model = GRUNet(embedding_matrix)
    model.to(device)
        
    # Defining loss function and optimizer
    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    
    model.train()
    print("Starting Training of {} model".format('GRU'))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        avg_accuracy = 0.
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()
            
            out, h = model(x.to(device), h)
            #out, h = model(x.to(device))
            loss = criterion(out[:,-1], label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()                    
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}"\
                      .format(epoch, counter, len(train_loader), \
                              round(avg_loss/counter, 2)))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}"\
              .format(epoch, EPOCHS, round(avg_loss/len(train_loader),2)))
        print("Total Time Elapsed: {} seconds".format(str(round(current_time - start_time,2))))
        epoch_times.append(round(current_time - start_time,2))
    print("Total Training Time: {} seconds".format(str(round(sum(epoch_times),2))))
    return model, optimizer
def score(model, test_x, test_y):
    pred = model.predict(torch.LongTensor(test_x).to(device).long())
    return round(roc_auc_score(torch.tensor(test_y).cpu().detach().numpy(),pred.cpu().detach().numpy()), 2)
model_gru, optimizer = train_model(train_loader)
index = sample(range(len(xvalid_pad)),500)
score(model_gru, xvalid_pad[index], yvalid[index])
checkpoint = {'model': model_gru,
              'state_dict': model_gru.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, '/kaggle/working/checkpoint.pth')
# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = checkpoint['model']
#     model.load_state_dict(checkpoint['state_dict'])
#     for parameter in model.parameters():
#         parameter.requires_grad = False

#     model.eval()
#     return model

# model = load_checkpoint('/kaggle/working/checkpoint.pth')

#del model_gru
#import gc
#torch.cuda.empty_cache()
#del model
#score(model, xvalid_pad[index], yvalid[index])
#%time model_gru.save('kaggle/working/model_gru.h5')