# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install transformers



import time

import sys

import copy

import torch 

import numpy as np

from scipy.sparse import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import pyarrow as pa



import torch.nn as nn

from torch.optim import lr_scheduler

import torch.nn.functional as F

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset,DataLoader

from transformers import DistilBertConfig,DistilBertTokenizer,DistilBertModel



import pandas as pd

from sklearn.preprocessing import LabelBinarizer



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
data = pd.read_csv('../input/vn-sentiment/data_sure - data.csv')
data
data['new'] = list(pd.get_dummies(data['nam_rate']).get_values())
data
## Feature engineering to prepare inputs for BERT....

# Y = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].astype(float)

# X = train['comment_text']



X = data['comment']

Y = data['new']



train, test = train_test_split(data, stratify=data['nam_rate'], test_size=0.3, random_state=42)
X_train = train['comment'].values

X_test = test['comment'].values

y_train = train['nam_rate'].values - 1

y_test = test['nam_rate'].values - 1 

# y_train_new = np.zeros((len(y_train), 5))

# y_test_new = np.zeros((len(y_test), 5))

y_train
# for i, y in enumerate(y_train):

#     y_train_new[i] = y

    

# for i, y in enumerate(y_test):

#     y_test_new[i] = y
def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):

    "Compute accuracy when `y_pred` and `y_true` are the same size."

    if sigmoid: y_pred = y_pred.sigmoid()

#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()

#Expected object of scalar type Bool but got scalar type Double for argument #2 'other'
config = DistilBertConfig(vocab_size=30522, dim=768,dropout=0.1,num_labels=5, n_layers=15, n_heads=12, hidden_dim=3072)
class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.num_labels = config.num_labels



        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.seq_classif_dropout)



        nn.init.xavier_normal_(self.classifier.weight)



    def forward(self, input_ids=None, attention_mask=None, head_mask=None, labels=None):

        distilbert_output = self.distilbert(input_ids=input_ids,

                                            attention_mask=attention_mask,

                                            head_mask=head_mask)

        hidden_state = distilbert_output[0]                    

        pooled_output = hidden_state[:, 0]                   

        pooled_output = self.pre_classifier(pooled_output)   

        pooled_output = nn.ReLU()(pooled_output)             

        pooled_output = self.dropout(pooled_output)        

        logits = self.classifier(pooled_output) 

        return F.softmax(logits)
max_seq_length = 120

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')





class text_dataset(Dataset):

    def __init__(self,x,y, transform=None):

        

        self.x = x

        self.y = y

        self.transform = transform

        

    def __getitem__(self,index):

        

        tokenized_comment = tokenizer.tokenize(self.x[index])

        

        if len(tokenized_comment) > max_seq_length:

            tokenized_comment = tokenized_comment[:max_seq_length]

            

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_comment)



        padding = [0] * (max_seq_length - len(ids_review))

        

        ids_review += padding

        

        assert len(ids_review) == max_seq_length

        

        #print(ids_review)

        ids_review = torch.tensor(ids_review)

        

        hcc = self.y[index] # toxic comment        

#         list_of_labels = [torch.from_numpy(hcc)]

        

        

        return ids_review, hcc

    

    def __len__(self):

        return len(self.x)

 
X_train[3]
tokenizer.tokenize(X_train[3])

text_dataset(X_train, y_train).__getitem__(6)[1]   ### Testing index 6 to see output
batch_size = 32





training_dataset = text_dataset(X_train,y_train)



test_dataset = text_dataset(X_test,y_test)



dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False),

                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                   }

dataset_sizes = {'train':len(X_train),

                'val':len(X_test)}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = DistilBertForSequenceClassification(config)

model.to(device)



print(device)

from tqdm import *

from sklearn.metrics import f1_score

def train_model(model, criterion, optimizer, scheduler, num_epochs=2):

    

    train_losses = []

    valid_losses = []

    avg_train_losses = []

    avg_valid_losses = []

    train_lost_iter = []

    valid_lost_iter = []

    best_val_score = 1000



    dataloader_train = dataloaders_dict['train']

    dataloader_valid = dataloaders_dict['val']

    

    for epoch in range(1, num_epochs + 1):

        model.train()

        train_pre = []

        valid_pre = []



        with tqdm(total=len(dataloader_train)) as pbar:

            for step, (ids_review, hcc) in enumerate(dataloader_train):

                optimizer.zero_grad()

                ids_review = ids_review.to(device)

                hcc = hcc.to(device)

                outputs = model(ids_review)

                train_pre += list(np.argmax(outputs.cpu().detach().numpy(), axis=1))

                loss = criterion(outputs, hcc.long())

                loss.backward()

                optimizer.step()

                train_losses.append(loss.item())

                train_lost_iter.append(loss.item())

                pbar.update()



        model.eval()

        with tqdm(total=len(dataloader_valid)) as pbar:

            for step, (ids_review, hcc) in enumerate(dataloader_valid):

                ids_review = ids_review.to(device)

                hcc = hcc.to(device)

                outputs = model(ids_review)

                loss = criterion(outputs, hcc.long())

                valid_losses.append(loss.item())

                valid_lost_iter.append(loss.item())

                valid_pre += list(np.argmax(outputs.cpu().detach().numpy(), axis=1))



                pbar.update()

                

        train_pre = np.array(train_pre).reshape(-1,)

        valid_pre = np.array(valid_pre).reshape(-1,)

        train_f1 = f1_score(y_train, train_pre, average='weighted')

        valid_f1 = f1_score(y_test, valid_pre, average='weighted')

        print(train_f1, valid_f1)



        

        scheduler.step()

        # print training/validation statistics 

        # calculate average loss over an epoch

        train_loss = np.average(train_losses)

        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)

        avg_valid_losses.append(valid_loss)

        

        if valid_f1 < best_val_score:

            best_val_score = valid_f1

            torch.save(model.state_dict(), 'distilbert_model_weights.pth')

            

        epoch_len = len(str(num_epochs))        

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +

                     f'train_loss: {train_loss:.5f} ' +

                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch

        train_losses = []

        valid_losses = []

        

    return model

 

print('done')
lrlast = .001

lrmain = 3e-5

#optim1 = torch.optim.Adam(

#    [

#        {"params":model.parameters,"lr": lrmain},

#        {"params":model.classifier.parameters(), "lr": lrlast},

#       

#   ])



optim1 = torch.optim.Adam(model.parameters(),lrmain)



optimizer_ft = optim1

criterion = nn.CrossEntropyLoss()



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.4)
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=21)