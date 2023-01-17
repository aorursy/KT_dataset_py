import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import os

import time

import torch

import torch.nn as nn

import torch.nn.functional as F



from sklearn import metrics

from sklearn.model_selection import train_test_split

import transformers

from transformers import AdamW



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



import warnings

warnings.filterwarnings("ignore")
t0 = torch.randn(2, 2, device=xm.xla_device()) #creating a tensor and sending it to the TPU

t1 = torch.randn(2, 2, device=xm.xla_device()) #creating a tensor and sending it to the TPU

print(t0 + t1) # As both tensors are now on the same device  i.e same TPU core we can perform any calculation on them like addition

print(t0.mm(t1)) # Matrix Multiplication
l_in = torch.randn(10, device=xm.xla_device())

linear = torch.nn.Linear(10, 20).to(xm.xla_device())

l_out = linear(l_in) #NOTE THAT THE TENSOR AND MODEL BOTH BE SENT TO THE DEVICE AS WE DID WITH GPUS , THEN ONLY we CAN PERFORM ANY OPERATION

print(l_out)
class config:

    

    MAX_LEN = 224

    TRAIN_BATCH_SIZE = 32

    VALID_BATCH_SIZE = 8

    EPOCHS = 1

    MODEL_PATH = "model.bin"

    TRAINING_FILE = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'

    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case =True)
class BERTDataset(torch.utils.data.Dataset):

    def __init__(self,text,target):

        self.text = text

        self.target = target

        self.tokenizer = config.TOKENIZER

        self.max_len = config.MAX_LEN 



    def __len__(self):

        return len(self.text)



    def __getitem__(self,idx):

        text  = str(self.text[idx])

        text = " ".join(text.split())



        inputs = self.tokenizer.encode_plus(

            text,

            None,

            add_special_tokens = True,

            max_length = self.max_len,

            pad_to_max_length = True

        )



        ids = inputs["input_ids"]

        mask = inputs["attention_mask"]



        return {

            'ids': torch.tensor(ids,dtype=torch.long),

            'mask': torch.tensor(mask,dtype=torch.long),

            'targets': torch.tensor(self.target[idx],dtype=torch.long)

        }
class BERTBaseUncased(nn.Module):

    def __init__(self):

        super(BERTBaseUncased,self).__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.bert_drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(768,1)



    def forward(self,ids,mask):

        _,o2 = self.bert(

            ids,

            mask

        )

        bo = self.bert_drop(o2)

        out = self.fc1(bo)



        return out
def loss_fn(outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
def eval_fn(data_loader, model, device):

    model.eval()

    fin_targets = []

    fin_outputs = []

    with torch.no_grad():

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            mask = d["mask"]

            targets = d["targets"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            outputs = model(

                ids=ids,

                mask=mask

            )

            fin_targets.extend(targets.cpu().detach().numpy().tolist())

            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets
train = pd.read_csv(config.TRAINING_FILE).fillna("none").sample(n=4000)

train_df, valid_df, train_tar, valid_tar = train_test_split(train.comment_text, train.toxic, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
def train_fn(data_loader, model, optimizer, device, scheduler,epoch,num_steps):

    model.train()



    for bi, d in enumerate(data_loader):

        ids = d["ids"]

        mask = d["mask"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        

        optimizer.zero_grad()

        outputs = model(

            ids=ids,

            mask=mask

        )



        loss = loss_fn(outputs, targets)

        loss.backward()

        #--------------------------------#------------------------#----------------------------#--------------------------#

        ####################################### CHANGE HAPPENS HERE #######################################################

        xm.optimizer_step(optimizer,barrier=True)

        ###################################################################################################################

        #-------------------------------#------------------------#----------------------------#---------------------------#

        if scheduler is not None:

                scheduler.step()

    

        

        if (bi+1) % 10 == 0:

            print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 

                   .format(epoch+1, 1, bi+1,num_steps, loss.item()))
def run():

    train_dataset = BERTDataset(

        text=train_df.values,

        target=train_tar.values

    )

    

    train_sampler = torch.utils.data.distributed.DistributedSampler(

          train_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=True)



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=4

    )



    valid_dataset = BERTDataset(

        text=valid_df.values,

        target=valid_tar.values

    )

    

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

          valid_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=False)



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=64,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=4

    )

    

    #-----------------------------#---------------------#-----------------------------------#-----------------------------------#

    ##################################### Change occurs Here ####################################################################



    device = xm.xla_device()

    model = BERTBaseUncased()

    model.to(device)

    

    #############################################################################################################################

    #----------------------------#----------------------#------------------------------------#-----------------------------------#

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    

    lr = 3e-5 * xm.xrt_world_size()    #You can or cannot make this change , it will work if not multiplied with xm.xrt_world_size()



    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=lr)



    best_accuracy = 0

    for epoch in range(config.EPOCHS):

        train_fn(train_data_loader, model, optimizer, device, scheduler=None,epoch=epoch,num_steps=num_train_steps)

        

        outputs, targets = eval_fn(valid_data_loader, model, device)

        

        outputs = np.array(outputs) >= 0.5

        accuracy = metrics.roc_auc_score(targets, outputs)

        print(f"AUC_SCORE = {accuracy}")

        if accuracy > best_accuracy:

            xm.save(model.state_dict(), config.MODEL_PATH)

            best_accuracy = accuracy
run()
!export XLA_USE_BF16=1
def train_fn(data_loader, model, optimizer, device, scheduler,epoch,num_steps):

    model.train()



    for bi, d in enumerate(data_loader):

        ids = d["ids"]

        mask = d["mask"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        

        optimizer.zero_grad()

        outputs = model(

            ids=ids,

            mask=mask

        )



        loss = loss_fn(outputs, targets)

        loss.backward()

        #--------------------------------#------------------------#----------------------------#--------------------------#

        ####################################### CHANGE HAPPENS HERE #######################################################

        xm.optimizer_step(optimizer)

        ###################################################################################################################

        #-------------------------------#------------------------#----------------------------#---------------------------#

        if scheduler is not None:

                scheduler.step()

    

        

        if (bi+1) % 10 == 0:

            print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 

                   .format(epoch+1, 1, bi+1,num_steps, loss.item()))
model = BERTBaseUncased()
def _run():

    

    train_dataset = BERTDataset(

        text=train_df.values,

        target=train_tar.values

    )

    

    train_sampler = torch.utils.data.distributed.DistributedSampler(

          train_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=True)



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=4

    )



    valid_dataset = BERTDataset(

        text=valid_df.values,

        target=valid_tar.values

    )

    

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

          valid_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=False)



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=64,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=4

    )



    device = xm.xla_device()

    model.to(device)

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]

    

    lr = 0.4 * 1e-5 * xm.xrt_world_size()



    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=lr)



    best_accuracy = 0

    #---------------------------------------#--------------------------------#----------------------------#-------------------------------#

    ########################################## Change occur In this Loop #################################################################

    for epoch in range(config.EPOCHS):

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=None,epoch=epoch,num_steps=num_train_steps)

        

        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        outputs, targets = eval_fn(para_loader.per_device_loader(device), model, device)

        

    ########################################################################################################################################

    #---------------------------------------#---------------------------------#------------------------------#------------------------------#

        

        outputs = np.array(outputs) >= 0.5

        accuracy = metrics.roc_auc_score(targets, outputs)

        print(f"AUC_SCORE = {accuracy}")

        if accuracy > best_accuracy:

            xm.save(model.state_dict(), config.MODEL_PATH)

            best_accuracy = accuracy
def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = _run()



FLAGS={}

#xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')