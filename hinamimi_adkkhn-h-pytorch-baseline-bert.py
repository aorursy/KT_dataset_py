from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from datetime import datetime

from pytz import timezone

datetime.now(timezone('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')



def refer_args(x):

    if type(x) == 'method':

        print(*x.__code__.co_varnames.split(), sep='\n')

    else:

        print(*[x for x in dir(x) if not x.startswith('__')], sep='\n')
from IPython.display import Image



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams 

import seaborn as sns

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torch.optim import Adam



def matplotlib_config():

    rcParams['font.family'] = 'sans-serif'

    rcParams['font.sans-serif'] = [

        'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao',

        'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP'

    ]

    rcParams['figure.figsize'] = 12, 8

    rcParams["font.size"] = 12



matplotlib_config()
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')



df_train.head()
df_train, df_val = train_test_split(df_train, train_size=0.8)
tokenizer = torch.hub.load(

    'huggingface/pytorch-transformers',

    'tokenizer',

    'bert-base-uncased'

)

config = torch.hub.load(

    'huggingface/pytorch-transformers',

    'config',

    'bert-base-uncased'

)

bert = torch.hub.load(

    'huggingface/pytorch-transformers',

    'model',

    'bert-base-uncased',

    config=config

)
class NLP_with_Disaster_Tweets_Dataset(Dataset):

    def __init__(self, df, transforms=None, mode='test'):

        self.raw_texts = self.texts = df['text'].tolist()

        if transforms is not None:

            self.texts = transforms(self.texts)

        self.mode = mode

        if mode != 'test':

            self.labels = df['target'].values

    

    def __iter__(self):

        if self.mode != 'test':

            return zip(self.texts, self.labels)

        else:

            return iter(self.texts)

    

    def __list__(self):

        return list(self.__iter__)

    

    def __len__(self):

        return len(self.texts)

    

    def __getitem__(self, idx):

        if self.mode != 'test':

            return self.texts[idx], self.labels[idx]

        else:

            return self.texts[idx]



def transforms(texts):

    embeds = []    

    for text in texts:

        embed = tokenizer.encode(

            text,                      

            add_special_tokens = True, 

            max_length = 160,           

            pad_to_max_length = True,

            return_tensors = 'pt'  

        )

        embeds.append(embed)    

    embeds = torch.cat(embeds, dim=0)

    return embeds



train_dataset = NLP_with_Disaster_Tweets_Dataset(df_train, transforms=transforms, mode='train')

train_dataloader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    drop_last=False

)



val_dataset = NLP_with_Disaster_Tweets_Dataset(df_val, transforms=transforms, mode='val')

val_dataloader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    drop_last=False

)



test_dataset = NLP_with_Disaster_Tweets_Dataset(df_test, transforms=transforms, mode='test')

test_dataloader = DataLoader(

    test_dataset,

    batch_size=32,

    shuffle=False,

    drop_last=False

)



train_size = len(train_dataset)

val_size = len(val_dataset)

test_size = len(test_dataset)
class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        self.bert = bert

        self.linear = nn.Linear(768, 1)

        # initialing weights and bias

        nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, inputs, **kwargs):

        vec, cls = self.bert(inputs, **kwargs)

        out = self.linear(cls)

        out = out.view(-1)

        return torch.sigmoid(out)



classifier = BertClassifier()

criterion = nn.BCELoss()

optimizer = Adam(

    classifier.parameters(),

    lr=0.001,

    betas=(0.9, 0.999),

    eps=1e-08,

    weight_decay=0,

    amsgrad=False

)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classifier = classifier.to(device)
train_losses = []

train_accs = []

train_preds = []



val_losses = []

val_accs = []

val_preds = []



num_epoch = 2

for epoch in range(num_epoch):

    train_loss = train_acc = 0

    train_pred = torch.tensor([]).to(device)

    _ = classifier.train()

    _ = torch.set_grad_enabled(True)

    loop = tqdm(train_dataloader)

    for x, y in loop:

        x, y = x.to(device), y.to(device)

        

        optimizer.zero_grad()

        z = classifier(x, attention_mask=(x > 0))

        loss = criterion(z, y.float())

        loss.backward()

        optimizer.step()

        

        train_pred = torch.cat([train_pred, torch.round(z)])

        train_loss += loss.item()

        acc = (torch.round(z) == y).sum()

        train_acc += acc.item()

        

        loop.set_description(f'Epoch {epoch+1}/{num_epoch}')

        loop.set_postfix(loss=f'{train_loss/train_size:.5f}', acc=f'{train_acc/train_size:.5f}')

    

    train_losses.append(train_loss/train_size)

    train_accs.append(train_acc/train_size)

    train_preds.append(train_pred.to('cpu').detach().numpy())

    

    val_loss = val_acc = 0

    val_pred = torch.tensor([]).to(device)

    _ = classifier.eval()

    _ = torch.set_grad_enabled(False)

    loop = val_dataloader

    for x, y in loop:

        x, y = x.to(device), y.to(device)

        

        with torch.no_grad():

            z = classifier(x, attention_mask=(x > 0))

            loss = criterion(z, y.float())

        val_loss += loss.item()

        acc = (torch.round(z) == y).sum()

        val_acc += acc.item()

        val_pred = torch.cat([val_pred, torch.round(z)])

        

        

    print(f'loss={val_loss/val_size:.5f}, acc={val_acc/val_size:.5f}')

    

    val_losses.append(val_loss/val_size)

    val_accs.append(val_acc/val_size)

    val_preds.append(val_pred.to('cpu').detach().numpy())

    
test_size = len(test_dataset)

test_loss = test_acc = 0

test_pred = torch.tensor([]).to(device)



_ = classifier.eval()

_ = torch.set_grad_enabled(False)

loop = tqdm(test_dataloader)

for x in loop:

    x = x.to(device)

    with torch.no_grad():

        z = classifier(x, attention_mask=(x > 0))

    test_pred = torch.cat([test_pred, torch.round(z)])



test_pred = test_pred.to('cpu').detach().numpy()
submission = pd.DataFrame()

submission['id'] = df_test['id']

submission['target'] = list(map(int, test_pred))

submission.to_csv('submission.csv', index=False)