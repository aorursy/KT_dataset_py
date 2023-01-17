# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch

from torch.utils.data import Dataset

from transformers import BertTokenizer

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

import torch.utils.data as Data



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

path = '/kaggle/input/ml2019fall-hw5'

trainx_path = os.path.join(path, 'train_x.csv')

trainy_path = os.path.join(path, 'train_y.csv')

testx_path = os.path.join(path, 'test_x.csv')

trainx = pd.read_csv(trainx_path)

trainy = pd.read_csv(trainy_path)

testx = pd.read_csv(testx_path)

trainx['comment'].iloc[0]

trainy.head()
class MyDataset(Dataset):

    def __init__(self, mode, dataX, dataY, tokenizer):

        self.mode = mode

        self.tokenizer = tokenizer

        self.dataX = dataX

        self.dataY = dataY

        self.len = len(dataX)

    

    def __getitem__(self, index):

        if self.mode == 'train':

            text = self.dataX['comment'].iloc[index]

            item = self.dataY['label'].iloc[index]

            tokens = self.tokenizer.tokenize(text)

            ids = self.tokenizer.convert_tokens_to_ids(tokens)

            return (ids, item)

        elif self.mode == 'test':

            text = self.dataX['comment'].iloc[index]

            item = None

            tokens = self.tokenizer.tokenize(text)

            ids = self.tokenizer.convert_tokens_to_ids(tokens)

            return (torch.tensor(ids), item)

    def __len__(self):

        return self.len



trainData = MyDataset('train', trainx[:10000], trainy[:10000], tokenizer)

valData = MyDataset('train', trainx[10000:], trainy[10000:], tokenizer)

testData = MyDataset('test', testx, None, tokenizer)
def create_mini_batch(samples):

    tokens_tensors = [torch.tensor(s[0]) for s in samples]

    labels_tensors = [torch.tensor(s[1]) for s in samples]

    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)    

    return tokens_tensors,torch.tensor(labels_tensors)

def create_mini_batch2(samples):

    tokens_tensors = [torch.tensor(s[0]) for s in samples]

    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)    

    return tokens_tensors

BATCH_SIZE = 32

trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)



# valDataLoader = DataLoader(trainData[10000:], batch_size=BATCH_SIZE, collate_fn=create_mini_batch)



testDataLoader = DataLoader(testData, batch_size=256, collate_fn=create_mini_batch2)
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)

model
def get_predictions(model, dataloader, compute_acc=False):

    predictions = None

    correct = 0

    total = 0

      

    with torch.no_grad():

        # 遍巡整個資料集

        for data in dataloader:

            # 將所有 tensors 移到 GPU 上

            if next(model.parameters()).is_cuda:

                data = [t.to("cuda:0") for t in data if t is not None]

            

            



            tokens_tensors = data[0]

            outputs = model(input_ids=tokens_tensors)

            

            logits = outputs[0]

            _, pred = torch.max(logits.data, 1)

            

            # 用來計算訓練集的分類準確率

            if compute_acc:

                labels = torch.tensor(data[1])

                total += labels.size(0)

                correct += (pred == labels).sum().item()

                

            # 將當前 batch 記錄下來

            if predictions is None:

                predictions = pred

            else:

                predictions = torch.cat((predictions, pred))

    

    if compute_acc:

        acc = correct / total

        return predictions, acc

    return predictions

    

# 讓模型跑在 GPU 上並取得訓練集的分類準確率

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device:", device)

model = model.to(device)

_, acc = get_predictions(model, valDataLoader, compute_acc=True)

print("classification acc:", acc)
import torch.nn.functional as F

model.train()



# 使用 Adam Optim 更新整個分類模型的參數

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)





EPOCHS = 6  # 幸運數字

for epoch in range(EPOCHS):

    

    running_loss = 0.0

    for data in trainDataLoader:

        

        tokens_tensors,labels = [t.to(device) for t in data]



        # 將參數梯度歸零

        optimizer.zero_grad()

        

        # forward pass

        outputs = model(input_ids=tokens_tensors, labels=labels)



        loss = outputs[0]

        

        # backward

        loss.sum().backward()

        optimizer.step()





        # 紀錄當前 batch loss

        running_loss += loss.sum().item()   

    # 計算分類準確率

    torch.save(model.state_dict(), './model{0}.pkl'.format(epoch))

    _, acc = get_predictions(model, valDataLoader, compute_acc=True)

    print('[epoch %d] loss: %.3f, acc: %.3f' %

          (epoch + 1, running_loss, acc))
def eval(model, testloader, compute_acc=False):

    predictions = None

    correct = 0

    total = 0

    model.eval()

    with torch.no_grad():

        # 遍巡整個資料集

        result = []

        for data in testloader:

            # 將所有 tensors 移到 GPU 上

#             if next(model.parameters()).is_cuda:

#                 data = [t.to("cuda:0") for t in data if t is not None]

#             print()

            

            tokens_tensors = data.to(device)

            outputs = model(input_ids=tokens_tensors)

            

            logits = outputs[0]

            _, pred = torch.max(logits.data, 1)

            for i in pred:

                result.append(int(i))

#                 print(int(i))

    return result



result = eval(model, testDataLoader)

df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})

output_path = ''

df
df.to_csv('res.csv', index=False)

os.listdir()