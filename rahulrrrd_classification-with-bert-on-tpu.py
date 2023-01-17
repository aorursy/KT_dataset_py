!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from sklearn import metrics

from sklearn import model_selection



from tqdm.notebook import tqdm



import torch

import torch.nn as nn

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup





import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils

import warnings



warnings.filterwarnings("ignore")

BERT_PATH = '../input/bert-base-uncased'

TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

MAX_LEN = 192

BATCH_SIZE = 128

V_BATCH_SIZE = 32

TRAIN_PATH = '../input/ag-news-classification-dataset/train.csv'

EPOCHS = 5

# MODEL_PATH = ''

#class to fetch a row from dataframe one by one and return dataset in well defined format



class prepare_dataset():

    def __init__(self, text, label):

        

        self.text = text

        self.label = label

        self.tokenizer = TOKENIZER

        self.max_len = MAX_LEN

        

    def __len__(self):

        return len(self.text)

    

    def __getitem__(self, idx):

        

        text = self.text[idx]

        text = " ".join(text.split())

        label = self.label[idx]

        

        #Torkenize the text

        

        inputs = self.tokenizer.encode_plus(text , None,

                                           add_special_tokens=True,

                                           max_length = self.max_len,

                                           pad_to_max_length=True)

        

        ids = inputs['input_ids']

        mask = inputs['attention_mask']

        token_type_ids = inputs['token_type_ids']

        

        padding_length = self.max_len - len(ids)

        

        

        #pad the tokenized vectors so that each has the same length of 192

        

        

        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

        

        return {

            'ids' : torch.tensor(ids, dtype=torch.long),

            'masks' : torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'targets': torch.tensor(label, dtype=torch.float)

        }

        

        
#class to load the model



class BertBaseUncased(nn.Module):

    def __init__(self):

        super(BertBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)

        self.dropout = nn.Dropout(0.2)

        

        

        # 4 output units in last layer since there are 4 classes 

        self.out = nn.Linear(768,4)                      

        

    def forward(self,ids, masks, token_type_ids):

        _, out = self.bert(ids, masks, token_type_ids)

        out = self.dropout(out)

        out = self.out(out)

        

        return out      




# function to calculate loss

# BCE with logitsloss is used because model will output logits [1.2, 0.7, 4.3, 2.3]

# BCE with logiloss will apply sigmoid to these outputs and calculate the BCE loss

def loss_fn(outputs, targets):

    return nn.BCEWithLogitsLoss()(outputs,targets)





def train_loop(dataloader, model ,optimizer, device,scheduler):

    

    model.train()

    

    epoch_loss = 0 

    

    counter = 0 

    for idx, batch in enumerate(dataloader):

        counter +=1

        ids = batch['ids']

        masks = batch['masks']

        token_type_ids =  batch['token_type_ids']

        targets = batch['targets']

         

        #move data to the accelerator device  GPU or TPU

        ids = ids.to(device, dtype=torch.long)

        masks = masks.to(device, dtype=torch.long)

        token_type_ids= token_type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype= torch.float)

        

        

        optimizer.zero_grad()

        

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)

        

        loss = loss_fn(outputs,targets)

        loss.backward()   

        

        xm.optimizer_step(optimizer)

#         optimizer.step()



        scheduler.step()

    

    

        if idx %50 == 0:

            xm.master_print(f'Batch: {idx} train_loss: {loss.item()}')

        

        epoch_loss+=loss.item()

        



        

    return epoch_loss/counter

        

        

        

        

        

def eval_loop(dataloader, model , device):

    

    model.eval()  

    epoch_acc = 0

    epoch_loss= 0

    counter = 0 

    for idx, batch in enumerate(dataloader):

        counter +=1

        ids = batch['ids']

        masks = batch['masks']

        token_type_ids =  batch['token_type_ids']

        targets = batch['targets']

        

        ids = ids.to(device, dtype=torch.long)

        masks = masks.to(device, dtype=torch.long)

        token_type_ids= token_type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype= torch.float)

        

        

        

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)

        

        loss = loss_fn(outputs,targets)

        

        #get the index of the maximum value 

        outputs = torch.argmax(outputs,axis=1)

        targets = torch.argmax(targets,axis=1)



        #calulate the accracy score

        acc = metrics.accuracy_score(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy())

        

        epoch_acc+=acc

        epoch_loss+= loss.item()

    

    final_acc = epoch_acc/counter

    epoch_loss = epoch_loss/counter

    return final_acc, epoch_loss

        

        

        





#function to convert the integer classes to one hot encoded array

#example 2 -> [0, 0, 1, 0]



def ohe(df,target_col):

    

    encoded = pd.get_dummies(df.sort_values(by=[target_col])[target_col])

    

    df = df.join(encoded)

    

    return df

    

    
def train():

    

    df = pd.read_csv(TRAIN_PATH).fillna('None')

    

    #split the data into train and validation sets

    train, valid = model_selection.train_test_split(df, test_size = 0.15, random_state=42, stratify=df['Class Index'].values)

    

    train = train.reset_index(drop=True)

    valid = valid.reset_index(drop=True)

    

    #one hot encode the classes

    train= ohe(train, 'Class Index')

    valid = ohe(valid, 'Class Index')

    

    train_labels = train[train.columns[-4:]].values

    valid_labels = valid[valid.columns[-4:]].values

    

    

    train_data = prepare_dataset(text=train['Description'].values,

                                label=train_labels)

    

    valid_data = prepare_dataset(text=valid['Description'].values,

                                label=valid_labels)

    

    

    train_sampler = torch.utils.data.DistributedSampler(train_data,

                                                       num_replicas=xm.xrt_world_size(),

                                                       rank= xm.get_ordinal(),

                                                       shuffle=True)



    valid_sampler = torch.utils.data.DistributedSampler(valid_data,

                                                       num_replicas=xm.xrt_world_size(),

                                                       rank= xm.get_ordinal(),

                                                       shuffle=False)

    

    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,num_workers=4,sampler=train_sampler,drop_last=True)

    valid_dataloader = torch.utils.data.DataLoader(valid_data,batch_size=V_BATCH_SIZE,num_workers=4,sampler=valid_sampler,drop_last=True)

    

    

    

#     device= torch.device('cuda')

    

    

    device = xm.xla_device()

        



    model = BertBaseUncased()

    model.to(device)

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {

            "params": [

                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)

            ],

            "weight_decay": 0.001,

        },

        {

            "params": [

                p for n, p in param_optimizer if any(nd in n for nd in no_decay)

            ],

            "weight_decay": 0.0,

        },

    ]

    

    num_train_steps = int(len(train_data)/BATCH_SIZE/xm.xrt_world_size() * EPOCHS)

    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    

    lr = 1e-4 * xm.xrt_world_size()

    

    optimizer = AdamW(optimizer_parameters,lr=lr)

    scheduler = get_linear_schedule_with_warmup(

        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps

    )

    

    best_acc=0

    

    for epoch in range(EPOCHS):

        

        para_loader = pl.ParallelLoader(train_dataloader, [device])

        

        train_loss = train_loop(para_loader.per_device_loader(device),model=model, optimizer=optimizer,scheduler=scheduler,device=device)

        

        para_loader = pl.ParallelLoader(valid_dataloader, [device])

        

        val_acc, val_loss = eval_loop(para_loader.per_device_loader(device), model, device)

        

#         print(f"EPOCH: {epoch} train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")

        

        if val_acc > best_acc:

            torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},'best_model.bin')

            

            best_acc=val_acc

            

        

        xm.master_print(f'Epoch: {epoch+1} train_loss: {train_loss} val_loss: {val_loss} Accracy: {val_acc}')

            

            

        

        

        

    
def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = train()

    

    

Flags ={}

xmp.spawn(_mp_fn,args=(Flags,), nprocs=1,start_method='fork')