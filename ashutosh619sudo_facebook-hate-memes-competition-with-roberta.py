# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sn
train = pd.read_json("/kaggle/input/facebook-hateful-meme-dataset/data/train.jsonl",lines=True)
test = pd.read_json("/kaggle/input/facebook-hateful-meme-dataset/data/test.jsonl",lines=True)

val = pd.read_json("/kaggle/input/facebook-hateful-meme-dataset/data/dev.jsonl",lines=True)
train.head()
train["label"].value_counts().plot(kind="bar")
plt.figure(figsize=(10,6))
img = plt.imread(f"/kaggle/input/facebook-hateful-meme-dataset/data/img/42953.png")
plt.imshow(img)
def return_len(data):
    data = data.split()
    return len(data)
train["lensequence"] = train["text"].apply(return_len)
train.head()
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
!pip install pytorch-transformers
from pytorch_transformers import RobertaModel, RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = 2
config
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification(config)
def prepare_features(seq_1, max_seq_length = 30, 
             zero_pad = False, include_CLS_token = True, include_SEP_token = True):
    tokens_a = tokenizer.tokenize(seq_1)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)

    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask
prepare_features("Hey I am Ashutosh")
plt.figure(figsize=(10,6))
sn.distplot(train["lensequence"])
max_len = 30
class HateDataset(Dataset):
    def __init__(self,dataframe):
        self.df = dataframe
    
    def __getitem__(self,index):
        text = self.df.text[index]
        text, _ = prepare_features(text,max_seq_length=35,zero_pad=True)
        target = self.df.label[index]
        
        return text,target
    
    def __len__(self):
        return len(self.df)
train_ds = HateDataset(train)
val_ds = HateDataset(val)
# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 4}
train_dl = DataLoader(train_ds,**params)
val_dl = DataLoader(val_ds,**params)
train_ds.__getitem__(0)[0].shape
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-05
optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)
device= torch.device("cuda")
inp = train_ds.__getitem__(0)[0].cuda()
output = model(inp)[0]
print(output.shape)
data,label = next(iter(train_dl))
print(data,label)
from tqdm import tqdm_notebook
max_epochs = 5
model.to(device)

for epoch in tqdm_notebook(range(max_epochs)):
    print("EPOCH -- {} ".format(epoch))
    
    for i,(text,label) in enumerate(train_dl):
            train_loss= 0
            optimizer.zero_grad()
            
            text = text.squeeze(0)
            text = text.to(device)
            
            label = label.to(device)
            
            output = model(text)[0]
            
            _,predicted = torch.max(output,1)
            
            loss = loss_function(output,label)
            loss.backward()
            
            train_loss += loss.item()
            optimizer.step()
            
            print("Train loss:-- {}".format(train_loss))
            
            if i%100 == 0:
                correct = 0
                total = 0
                for text, label in val_dl:
                    text = text.squeeze(0)

                    text = text.to(device)
                    label = label.to(device)

                    output = model.forward(text)[0]
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted.cpu() == label.cpu()).sum()
                accuracy = 100.00 * correct.numpy() / total
                print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
            
            
