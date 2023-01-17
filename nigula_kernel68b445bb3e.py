# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import math

import numpy as np

import pandas as pd

from tqdm import tqdm

import time

import transformers

import re



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, roc_auc_score

import matplotlib.pyplot as plt
df1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
df2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
df2_cut = df2[['comment_text','toxic']]
df1_cut = df1[['comment_text','toxic']]
df1_cut.head()
df_train_toxic_rus = pd.read_csv("/kaggle/input/toxic-rus-train/toxic_rus_train.csv")

df_test_toxic_rus = pd.read_csv("/kaggle/input/toxic-rus-test/toxic_rus_test.csv")
df_train_toxic_rus.head()
# df_train_toxic_rus = df_train_toxic_rus[['text','toxic']]

df_train_toxic_rus.columns = ['comment_text','toxic']



# df_test_toxic_rus = df_test_toxic_rus[['text','toxic']]

df_test_toxic_rus.columns = ['comment_text','toxic']
df1_cut.head()
df = pd.concat([df_train_toxic_rus, df_test_toxic_rus])
df.info()
df.columns = ['comment_text','toxic']
df_eng = pd.concat([df2_cut,df1_cut])
df_eng.info()
df = pd.concat([df, df_eng], ignore_index = True)
df.iloc[14400:14430]
# df = df.sample(frac=1).reset_index(drop=True)
def clean(text):

#     text = text.fillna("fillna").str.lower()

    text = text.map(lambda x: re.sub('\\n',' ',str(x)))

    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))

    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))

    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))

    return text
df['comment_text'] = clean(df['comment_text'])
df_val = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

df_test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
df_val['comment_text'] = clean(df_val['comment_text'])

df_test['content'] = clean(df_test['content'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
pretrained_weights = 'xlm-roberta-large'
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(pretrained_weights)

xlm_model = transformers.XLMRobertaModel.from_pretrained(pretrained_weights).to(device)
def pad_or_cut(text,max_len,pad_index):

    text = text[:1500]

    tokenized_text = tokenizer.encode(text)

    if len(tokenized_text) >  max_len:

        tokenized_text = tokenized_text[:max_len]

    elif len(tokenized_text) <  max_len:

        tokenized_text += [pad_index] * (max_len - len(tokenized_text))

    return tokenized_text



def sentence_embedder_model(text_batch):

    pad_index = 1

    max_len = 256



    batch = torch.tensor([pad_or_cut(text,max_len,pad_index) for text in text_batch])

#     print(batch.shape)



    batch = batch[:, :((batch != pad_index).long()).sum(dim=1).max()].to(device)

#     print(batch.shape)

    pad_mask = (batch != pad_index).long().to(device)



#     print("xlm_model ...")

    with torch.no_grad():

        sequence_output, pooled_output = xlm_model(batch, attention_mask=pad_mask)

#     print("sequence_output", sequence_output.shape)

    sequence_lengths = (pad_mask).sum(dim=1)

    sequence_lengths[sequence_lengths == 0.] = 1



    pad_mask_output = pad_mask.unsqueeze(-1).repeat(1, 1, sequence_output.size(-1))



    sequence_output = sequence_output * pad_mask_output

#     print("sequence_output",sequence_output.shape)

    lengths_scaling = sequence_lengths.float() / sequence_output.size(1)

    lengths_scaling = lengths_scaling.unsqueeze(1).repeat((1, sequence_output.size(-1)))



    sequence_output = sequence_output.mean(dim=1)

#     print("sequence_output mean",sequence_output.shape)



    sequence_output = sequence_output / lengths_scaling#.to(sequence_output.device)



    norm = sequence_output.norm(dim=1).unsqueeze(1).repeat((1, sequence_output.size(-1)))



    sequence_output = sequence_output / norm



    return  torch.tensor(sequence_output.tolist()) #[float(t) for t in sequence_output]

# robert_vect = sentence_embedder_model(["как купить слона","hey"])

# robert_vect = sentence_embedder_model(["как купить слона"*10,"hey"]*32)

# robert_vect = sentence_embedder_model(["h"*5000])

# print(robert_vect.shape)
EPOCHS = 50

BATCH_SIZE = 64

LEARNING_RATE = 0.001
class RawDataset(Dataset):

    def __init__(self, text, target):

        self.text = text

        self.target = target

        

    def __len__(self):

        return len(self.text)

    

    def __getitem__(self, index):

        return self.text[index], self.target[index]
class binaryClassification(nn.Module):

    def __init__(self):

        super(binaryClassification, self).__init__()

        # Number of input features is 12.

        self.layer_1 = nn.Linear(1024, 64) 

        self.layer_2 = nn.Linear(64, 64)

        self.layer_out = nn.Linear(64, 1) 

        

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

        self.batchnorm1 = nn.BatchNorm1d(64)

        self.batchnorm2 = nn.BatchNorm1d(64)

        

    def forward(self, x):

        x = self.relu(self.layer_1(x))

        x = self.batchnorm1(x)

        x = self.relu(self.layer_2(x))

        x = self.batchnorm2(x)

        x = self.dropout(x)

        x = self.layer_out(x)

        

        return x
TRAIN_SPLIT = 200000

VAL_SPLIT = -1
train_data = RawDataset(list(df['comment_text'])[:TRAIN_SPLIT],list(df['toxic'])[:TRAIN_SPLIT])

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)



val_data = RawDataset(list(df_val['comment_text'])[:VAL_SPLIT],list(df_val['toxic'])[:VAL_SPLIT])

val_loader = DataLoader(dataset=val_data, batch_size=int(BATCH_SIZE/4))
model = binaryClassification()

model.to(device)

print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
def roc_auc(y_test,y_pred_tag):

    

    y_test = torch.round(y_test.detach().cpu())

    print("y_test", y_test[:10])

    

    

    y_pred_tag= y_pred_tag.detach().cpu()

    y_pred_tag = torch.sigmoid(y_pred_tag)

    print("y_pred_tag", y_pred_tag[:10])

    

    roc_auc = roc_auc_score(y_test, y_pred_tag)

    return roc_auc
# for instance in list(tqdm._instances):

#     tqdm._decr_instances(instance)

EPOCHS = 1

best_loss = 10000

best_roc_auc = 0

patience = 0

plot_every = int(len(train_loader)/3)

for e in range(1, EPOCHS+1):

    model.train()

    

    train_epoch_loss = []

#     train_epoch_roc_auc = []

    progress_bar = tqdm(total=len(train_loader), desc='Train')

    iteration = 0

    

    for instance in list(tqdm._instances):

        tqdm._decr_instances(instance)

    

    for X_batch, y_batch in train_loader:

        y_batch = y_batch.to(device)

#         print("sent embedder ...")

        X_batch = sentence_embedder_model(X_batch).to(device)

#         X_batch = torch.stack([sentence_embedder_model(sent) for sent in X_batch])

#         X_batch = torch.tensor(X_batch.numpy()).to(device)

        

        optimizer.zero_grad()

#         print("model forward ...")

        y_pred = model(X_batch)

        y_batch = y_batch.type_as(y_pred)

        loss = criterion(y_pred, y_batch.unsqueeze(1))

#         roc_auc_value = roc_auc(y_pred, y_batch.unsqueeze(1))

        

        loss.backward()

        optimizer.step()

        

        train_epoch_loss.append(loss.item())

#         train_epoch_roc_auc.append(roc_auc_value)

        progress_bar.update()

        progress_bar.set_postfix(epoch = e,loss=np.mean(train_epoch_loss[-100:]))#roc = np.mean(roc_auc_value[-100:])

        iteration += 1

        if iteration%plot_every == 0:

            plt.plot(train_epoch_loss)

            plt.ylabel('Loss')

            plt.show()

            

#     print(f'Epoch {e+0:03}: | Loss: {np.mean(train_epoch_loss):.5f} | Acc: {np.mean(train_epoch_f1):.3f}')

    

    #========================EVALUATION=============

    model.eval()

    eval_epoch_loss = []

    eval_epoch_preds = []

    eval_epoch_trues = []

#     eval_progress_bar = tqdm(total=len(val_loader), desc='Eval')

    print("EVALUATE IS RUNNING ....")

    for X_batch, y_batch in val_loader:

        y_batch = y_batch.to(device)

        

#         X_batch = torch.stack([sentence_embedder_model(sent) for sent in X_batch])

#         X_batch = torch.tensor(X_batch.numpy()).to(device)

        X_batch = sentence_embedder_model(X_batch).to(device)

        

        y_pred = model(X_batch)

        y_batch = y_batch.type_as(y_pred)

        

        ev_loss = criterion(y_pred, y_batch.unsqueeze(1))

        eval_epoch_loss.append(ev_loss.item())

        

        eval_epoch_preds.extend(y_pred)

        eval_epoch_trues.extend(y_batch.unsqueeze(1))

        

#         eval_progress_bar.update()

#         eval_progress_bar.set_postfix(eval_loss=np.mean(eval_epoch_loss[-100:]))

    

    eval_epoch_preds = torch.cat(eval_epoch_preds)

    eval_epoch_trues = torch.cat(eval_epoch_trues,0)

    roc_auc_value = roc_auc(eval_epoch_trues, eval_epoch_preds)

    

    mean_epoch_loss = np.mean(eval_epoch_loss)

    if mean_epoch_loss < best_loss and roc_auc_value > best_roc_auc:

        print("NEW BEST RESULT! SAVING ...")

        best_loss = mean_epoch_loss

        best_roc_auc = roc_auc_value

        patience = 0

#         torch.save(model.state_dict(),

#                        'train_results/model_state_dict.pth')

#         torch.save(optimizer.state_dict(),

#                        'train_results/optimizer_state_dict.pth')

    elif mean_epoch_loss >= best_loss and roc_auc_value <= best_roc_auc:

        patience += 1

    elif mean_epoch_loss >= best_loss:

        best_roc_auc = roc_auc_value

        patience += 1

    elif roc_auc_value <= best_roc_auc:

        best_loss = mean_epoch_loss

        patience += 1

    if patience > 3:

        print("out of patience!")

        break

        

    print(f'Epoch {e+0:03}: | Validation Loss: {mean_epoch_loss:.5f} | Validation roc_auc_value: {roc_auc_value:.3f}')

    print("="*100)

#     break
roc_auc_value
df_test = pd.read_csv(os.path.join("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/","test.csv"))

df_test.head()
class RawDataset_test(Dataset):

    def __init__(self, text):

        self.text = text

        

    def __len__(self):

        return len(self.text)

    

    def __getitem__(self, index):

        return self.text[index]
test_data = RawDataset_test(list(df_test['content']))

test_loader = DataLoader(dataset=test_data, batch_size=int(BATCH_SIZE/4))
model.eval()

test_epoch_preds = []

print("TEST IS RUNNING ....")

for X_batch in test_loader:



    X_batch = sentence_embedder_model(X_batch).to(device)



    y_pred = model(X_batch)



    test_epoch_preds.extend(y_pred)



test_epoch_preds = torch.cat(test_epoch_preds)

test_epoch_preds[:10]
test_epoch_preds_t= test_epoch_preds.detach().cpu()

test_epoch_preds_t = torch.sigmoid(test_epoch_preds_t)
final_test_preds = list(float(i) for i in test_epoch_preds_t)
final_test_preds[:10]
list(df_test.id)[:10]
my_submission = pd.DataFrame({'Id': list(df_test.id), 'toxic': final_test_preds})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
os.listdir("./")