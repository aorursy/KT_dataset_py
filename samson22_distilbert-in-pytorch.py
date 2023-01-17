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



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

test_labels = pd.read_csv('//kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

train = pd.read_csv('//kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('//kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')
train.head(7)
## Feature engineering to prepare inputs for BERT....





Y = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].astype(float)

X = train['comment_text']





X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
print('train_x shape is {}' .format({X_train.shape}))

print('test_x shape is {}' .format({X_test.shape}))

print('train_y shape is {}' .format({y_train.shape}))
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
def accuracy_thresh(y_pred, y_true, thresh:float=0.4, sigmoid:bool=True):

    "Compute accuracy when `y_pred` and `y_true` are the same size."

    if sigmoid: y_pred = y_pred.sigmoid()

#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()

#Expected object of scalar type Bool but got scalar type Double for argument #2 'other'
config = DistilBertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,dropout=0.1,num_labels=6,

        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.num_labels = config.num_labels



        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

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

        return logits
max_seq_length = 256

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

        list_of_labels = [torch.from_numpy(hcc)]

        

        

        return ids_review, list_of_labels[0]

    

    def __len__(self):

        return len(self.x)

 
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

def train_model(model, criterion, optimizer, scheduler, num_epochs=2):

    model.train()

    since = time.time()

    print('starting')

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 100



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            

            beta_score_accuracy = 0.0

            

            micro_roc_auc_acc = 0.0

            

            

            # Iterate over data.

            for inputs, hcc in dataloaders_dict[phase]:

                

                inputs = inputs.to(device) 

                hcc = hcc.to(device)

            

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    

                    loss = criterion(outputs,hcc.float())

                    

                    if phase == 'train':

                        

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() * inputs.size(0)

                

                micro_roc_auc_acc +=  accuracy_thresh(outputs.view(-1,6),hcc.view(-1,6))

                

                #print(micro_roc_auc_acc)



                

            epoch_loss = running_loss / dataset_sizes[phase]



            

            epoch_micro_roc_acc = micro_roc_auc_acc / dataset_sizes[phase]



            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))

            print('{} micro_roc_auc_acc: {:.4f}'.format( phase, epoch_micro_roc_acc))



            if phase == 'val' and epoch_loss < best_loss:

                print('saving with loss of {}'.format(epoch_loss),

                      'improved over previous {}'.format(best_loss))

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), 'distilbert_model_weights.pth')

         



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(float(best_loss)))



    # load best model weights

    model.load_state_dict(best_model_wts)

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

criterion = nn.BCEWithLogitsLoss()



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=8)
#y_test = test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

x_test = test['comment_text']

y_test = np.zeros(x_test.shape[0]*6).reshape(x_test.shape[0],6)



test_dataset = text_dataset(x_test,y_test)

prediction_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)



def preds(model,test_loader):

    predictions = []

    for inputs, sentiment in test_loader:

        inputs = inputs.to(device) 

        sentiment = sentiment.to(device)

        with torch.no_grad():

            outputs = model(inputs)

            outputs = torch.sigmoid(outputs)

            predictions.append(outputs.cpu().detach().numpy().tolist())

    return predictions
predictions = preds(model=model_ft1,test_loader=prediction_dataloader)

predictions = np.array(predictions)[:,0]
submission = pd.DataFrame(predictions,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])

test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]=submission

final_sub = test[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]

final_sub.head()
final_sub.to_csv('submissions.csv',index=False)#

final_sub.head()