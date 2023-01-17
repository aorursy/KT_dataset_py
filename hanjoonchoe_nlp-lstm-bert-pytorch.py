!pip install transformers
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

from transformers import BertTokenizer,BertModel

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset

from torch.nn.utils.rnn import pack_padded_sequence

from torch.optim import AdamW

from tqdm import tqdm

from argparse import ArgumentParser

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Accuracy, Loss

from ignite.engine.engine import Engine, State, Events

from ignite.handlers import EarlyStopping

from ignite.contrib.handlers import TensorboardLogger, ProgressBar

from ignite.utils import convert_tensor

from torch.optim.lr_scheduler import ExponentialLR

import warnings  

warnings.filterwarnings('ignore')
def readfiles():

    path = '/kaggle/input/nlp-getting-started'

    train = pd.read_csv(os.path.join(path,'train.csv'))

    test = pd.read_csv(os.path.join(path,'test.csv'))

    sample_subs = pd.read_csv(os.path.join(path,'sample_submission.csv'))

    

    return train,test,sample_subs



train,test,sample_subs = readfiles()
from transformers import BertTokenizer

def Bert_Tokenizer(model_name):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer

tokenizer = Bert_Tokenizer('bert-base-uncased')
class TextDataset(Dataset):

    def __init__(self,df,tokenizer,max_len):

        

        self.bert_encode = tokenizer

        self.texts = df.text.values

        self.labels = df.target.values

        self.max_len = max_len

        

    def __len__(self):

        

        return len(self.texts)

    

    def __getitem__(self,idx):

        

        tokens,mask,tokens_len = self.get_token_mask(self.texts[idx],self.max_len)

        label = self.labels[idx]

        return [torch.tensor(tokens),torch.tensor(mask),torch.tensor(tokens_len)],label

        

    def get_token_mask(self,text,max_len):

        

        tokens = []

        mask = []

        text = self.bert_encode.encode(text)

        size = len(text)

        pads = self.bert_encode.encode(['PAD']*(max(0,max_len-size)))

        tokens[:max(max_len,size)] = text[:max(max_len,size)]

        tokens = tokens + pads[1:-1]

        mask = [1]*size+[0]*len(pads[1:-1])

        tokens_len = len(tokens)

        

        return tokens,mask,tokens_len
def get_data_loaders():

    from sklearn.model_selection import train_test_split

    x_train , x_valid = train_test_split(train, test_size=0.1,random_state=2020)

    train_dataset = TextDataset(x_train,tokenizer=tokenizer,max_len=120)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)

    valid_dataset = TextDataset(x_valid,tokenizer=tokenizer,max_len=120)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=1,shuffle=True)

    

    return train_loader , valid_loader
class MixedBertModel(nn.Module):

    def __init__(self,pre_trained='bert-base-uncased'):

        super().__init__()

        

        self.bert = BertModel.from_pretrained(pre_trained)

        self.hidden_size = self.bert.config.hidden_size

        self.LSTM = nn.LSTM(self.hidden_size,self.hidden_size,bidirectional=True)

        self.clf = nn.Linear(self.hidden_size*2,1)

        

    def forward(self,inputs):

        

        encoded_layers, pooled_output = self.bert(input_ids=inputs[0],attention_mask=inputs[1])

        encoded_layers = encoded_layers.permute(1, 0, 2)

        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, inputs[2]))

        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)

        output_hidden = F.dropout(output_hidden,0.2)

        output = self.clf(output_hidden)

        

        return F.sigmoid(output)
def _prepare_batch(batch, device=None, non_blocking=False):



    x, y = batch

    return (convert_tensor(x, device=device, non_blocking=non_blocking),

            convert_tensor(y, device=device, non_blocking=non_blocking))

def create_supervised_trainer1(model, optimizer, loss_fn, metrics={}, device=None):



    def _update(engine, batch):

        model.train()

        optimizer.zero_grad()

        x, y = _prepare_batch(batch, device=device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y.float())

        loss.backward()

        optimizer.step()

        return loss.item(), y_pred, y



    def _metrics_transform(output):

        return output[1], output[2]



    engine = Engine(_update)



    for name, metric in metrics.items():

        metric._output_transform = _metrics_transform

        metric.attach(engine, name)



    return engine



def create_supervised_evaluator1(model, metrics=None,

                                device=None, non_blocking=False,

                                prepare_batch=_prepare_batch,

                                output_transform=lambda x, y, y_pred: (y_pred, y,)):



    metrics = metrics or {}



    if device:

        model



    def _inference(engine, batch):

        model.eval()

        with torch.no_grad():

            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

            y_pred = model(x)

            return output_transform(x, y.float(), y_pred)



    engine = Engine(_inference)



    for name, metric in metrics.items():

        metric.attach(engine, name)



    return engine
def run(log_interval=100,epochs=2,lr=0.000006):

    train_loader ,valid_loader = get_data_loaders()

    model = MixedBertModel()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



    criterion = nn.BCELoss()

    optimizer = AdamW(model.parameters(), lr=lr)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    trainer = create_supervised_trainer1(model.to(device), optimizer, criterion, device=device)

    evaluator = create_supervised_evaluator1(model.to(device), metrics={'BCELoss': Loss(criterion)}, device=device)



    if log_interval is None:

        e = Events.ITERATION_COMPLETED

        log_interval = 1

    else:

        e = Events.ITERATION_COMPLETED(every=log_interval)

        

    desc = "loss: {:.4f} | lr: {:.4f}"

    pbar = tqdm(

        initial=0, leave=False, total=len(train_loader),

        desc=desc.format(0, lr)

    )

    @trainer.on(e)

    def log_training_loss(engine):

        pbar.refresh()

        lr = optimizer.param_groups[0]['lr']

        pbar.desc = desc.format(engine.state.output[0], lr)

        pbar.update(log_interval)

        

    @trainer.on(Events.EPOCH_COMPLETED)

    def update_lr_scheduler(engine):

        lr_scheduler.step()

        



            

    @trainer.on(Events.EPOCH_COMPLETED)

    def log_training_results(engine):

        evaluator.run(train_loader)

        metrics = evaluator.state.metrics

        avg_loss = metrics['BCELoss']

        tqdm.write(

            "Train Epoch: {} BCE loss: {:.2f}".format(engine.state.epoch, avg_loss)

        )

    

    @trainer.on(Events.EPOCH_COMPLETED)

    def log_validation_results(engine):

        pbar.refresh()

        evaluator.run(valid_loader)

        metrics = evaluator.state.metrics

        avg_loss = metrics['BCELoss']

        tqdm.write(

            "Valid Epoch: {} BCE loss: {:.2f}".format(engine.state.epoch, avg_loss)

        )

        pbar.n = pbar.last_print_n = 0

    

    

    try:

        trainer.run(train_loader, max_epochs=epochs)

    

    except Exception as e:

        import traceback

        print(traceback.format_exc())

    return model
model =run()
class TestTextDataset(Dataset):

    def __init__(self,df,tokenizer,max_len):

        

        self.bert_encode = tokenizer

        self.texts = df.text.values

        self.max_len = max_len

        

    def __len__(self):

        

        return len(self.texts)

    

    def __getitem__(self,idx):

        

        tokens,mask,tokens_len = self.get_token_mask(self.texts[idx],self.max_len)

        return [torch.tensor(tokens),torch.tensor(mask),torch.tensor(tokens_len)]

        

    def get_token_mask(self,text,max_len):

        

        tokens = []

        mask = []

        text = self.bert_encode.encode(text)

        size = len(text)

        pads = self.bert_encode.encode(['PAD']*(max(0,max_len-size)))

        tokens[:max(max_len,size)] = text[:max(max_len,size)]

        tokens = tokens + pads[1:-1]

        mask = [1]*size+[0]*len(pads[1:-1])

        tokens_len = len(tokens)

        

        return tokens,mask,tokens_len
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model.eval()

predictions = []

test_dataset = TestTextDataset(test,tokenizer=tokenizer,max_len=120)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

with torch.no_grad():

    for idx , (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

        inputs = [a.to(device) for a in inputs]

        preds = model(inputs)

        predictions.append(preds.cpu().detach().numpy())

        

predictions = np.vstack(predictions)
sample_subs.target = np.round(np.vstack(predictions)).astype(int)

sample_subs.head(20)
sample_subs.to_csv('submission.csv', index = False)