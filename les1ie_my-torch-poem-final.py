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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
from random import shuffle
from tqdm import tqdm
import torchvision

from pathlib import Path
from tqdm import tqdm

from tqdm.notebook import tqdm

data_path = "../input/tangshi2020/tang.npz"


class Tang(Dataset):
    def __init__(self, npz_path, is_train=True, poem_len=48):
        self.data = np.load(npz_path, allow_pickle=True)
        print("Read {} finish.".format(npz_path))
        # print(self.data.files)
        self.poem_len = poem_len
        self.ix2word = self.data['ix2word'].item()
        self.word2ix = self.data['word2ix'].item()
        self.raw_poem = self.data['data']
        self.poem = [i for i in self.raw_poem.reshape(-1) if i != 8292]  # 压缩到1维，去掉空格 <START> <EOP>

    def print_poem(self, s):
        t = []
        for i in s:
            t.append(self.ix2word[i])
        print("".join(t))

    def __len__(self):
        return int(len(self.poem) / self.poem_len)
        # return len(self.poem)

    def __getitem__(self, idx):
        item = self.poem[idx * self.poem_len: (idx + 1) * self.poem_len]
        item = torch.from_numpy(np.array(item)).long()

        label = self.poem[idx * self.poem_len + 1: (idx + 1) * self.poem_len + 1]
        label = torch.from_numpy(np.array(label)).long()
        return item, label


    
class NewNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """

        :param vocab_size: len(ix2word)
        :param embedding_dim:
        :param hidden_dim:
        """
        super(NewNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True,dropout=0.3, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim,4096)
        self.fc2 = nn.Linear(4096,8192)
        self.fc3 = nn.Linear(8192,vocab_size)

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)   
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0)) 
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output,hidden
    
class Config:
    embedding_dim = 300
    hidden_dim = 2048
    lr = 0.001
    max_gen_len = 50
    epochs = 1
    batch_size = 128
    gamma = 0.1
    step_size = 10
    LSTM_layers=3


def prepare_data(path):
    datas = np.load(path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    data = torch.from_numpy(data)
    
    dataloader = DataLoader(data, batch_size=Config.batch_size, shuffle=False, )

    return dataloader, ix2word, word2ix


def train(dataloader, word2ix,model,epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)
    # loss_meter = meter.
    # loss =
    for epoch in range( epochs):
#         print("start epoch: ", epoch)
        train_loader = tqdm(dataloader)
        train_loader.set_description('epoch: {}/{} lr: {:.4f} '.format(epoch, epochs, scheduler.get_lr()[0]))

        for inputs, labels in train_loader:
            # for i, data in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            optimizer.zero_grad()
            output, hidden = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # _, pred = output.top(1)
            # prec1, prec2 = accuracy(output, labels, topk=(1, 2))
            # top1.update(prec1.item(), inputs.size(0))
            # train_loss += loss.item()
            # postfix = {"train_loss": "{:.6f}".format(train_loss / (i + 1)), "train_acc": "{:.6f}".format(top1.avg)}
        scheduler.step()  # 一个epoch 降低一次学习率


if __name__ == '__main__':
    data = Tang(npz_path=data_path)
    ix2word = data.ix2word
    word2ix = data.word2ix

    dataloader = DataLoader(data, batch_size=Config.batch_size, num_workers=0)
    model = NewNet(len(word2ix), embedding_dim=Config.embedding_dim, hidden_dim=Config.hidden_dim)
    
    train(  dataloader, word2ix,model, epochs=Config.epochs)


data,ix2word, word2ix = prepare_data(data_path)

train(  dataloader, word2ix,model, epochs=2)
def generate_final(model, start_words, ix2word, word2ix,device):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    
    #最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers*1,1,Config.hidden_dim),dtype=torch.float)
    input = input.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
            for i in range(48):#诗的长度
                output, hidden = model(input, hidden)
                # 如果在给定的句首中，input为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
               # 否则将output作为下一个input进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()#输出的预测的字
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)
                if w == '<EOP>': # 输出了结束标志就退出
                    del results[-1]
                    break
    return results

result = generate_final(model, "雪", ix2word, word2ix,"cuda")
"".join(result)
result = generate_final(model, "湖光秋月两相和", ix2word, word2ix,"cuda")
"".join(result)
result = generate_final(model, "忽如一夜春风来", ix2word, word2ix,"cuda")
"".join(result)
result = generate_final(model, "窗含西岭千秋雪，", ix2word, word2ix,"cuda")
"".join(result)
"".join( generate_final(model, "窗", ix2word, word2ix,"cuda"))
!nvidia-smi
torch.save(model.state_dict(), "poem.pt")
