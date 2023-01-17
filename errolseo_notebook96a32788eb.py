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
!pip install hb-config
import numpy as np

import pandas as pd

import wandb

import torch

from torch import nn

import torch.nn.functional as F

from transformers import AdamW, get_linear_schedule_with_warmup

import os

from hbconfig import Config
train_data = "/kaggle/input/g8wamyvdmxl7qyr/1.train.csv"

test_data = "/kaggle/input/g8wamyvdmxl7qyr/1.test.csv"



train_df = pd.read_csv(train_data)

test_df = pd.read_csv(test_data)



train_df.head()
config = Config



config.vocab_size = 8217

config.hidden_size = 512

config.emb_size = 256

config.lr = 5e-5

config.batch_size = 128

config.warmup = 1000

config.valid_data_rate = 0.08

config.epoch = 50

config.threshold = 0.5

config.max_len = 20
word_list = list(set(" ".join(train_df['sentence1'].append(train_df['sentence2'])).split()))

word_dict = {w: i+2 for i, w in enumerate(word_list)}

word_dict['<pad>'] = 0

word_dict['<unk>'] = 1 

word_dict['<eos>'] = 2

train_sen_a_bow = [[word_dict[word] for word in sen.split()] for sen in train_df['sentence1']]

train_sen_b_bow = [[word_dict[word] for word in sen.split()] for sen in train_df['sentence2']]

test_sen_a_bow = [[word_dict[word] if word in word_dict else 1 for word in sen.split()] for sen in test_df['sentence1']]

test_sen_b_bow = [[word_dict[word] if word in word_dict else 1 for word in sen.split()] for sen in test_df['sentence2']]

for i, sen in enumerate(train_sen_a_bow):

    sen = sen + [2]

    sen = sen + [0] * (config.max_len - len(sen))

    train_sen_a_bow[i] = sen

    

for i, sen in enumerate(train_sen_b_bow):

    sen = sen + [2]

    sen = sen + [0] * (config.max_len - len(sen))

    train_sen_b_bow[i] = sen

    

for i, sen in enumerate(test_sen_a_bow):

    sen = sen + [2]

    sen = sen + [0] * (config.max_len - len(sen))

    test_sen_a_bow[i] = sen

    

for i, sen in enumerate(test_sen_b_bow):

    sen = sen + [2]

    sen = sen + [0] * (config.max_len - len(sen))

    test_sen_b_bow[i] = sen

train_sen_a_bow = np.array(train_sen_a_bow)

train_sen_b_bow = np.array(train_sen_b_bow)

test_sen_a_bow = np.array(test_sen_a_bow)

test_sen_b_bow = np.array(test_sen_b_bow)
class STS_Dataset(torch.utils.data.Dataset):

    def __init__(self, sen_a, sen_b, label=None):

        self.sen_a = torch.tensor(sen_a, dtype=torch.long)

        self.sen_b = torch.tensor(sen_b, dtype=torch.long)

        self.is_labeled = False

        if isinstance(label, np.ndarray):

            self.label = torch.tensor(label, dtype=torch.float32)

            self.is_labeled = True

    

    def __len__(self):

        return len(self.sen_a)



    def __getitem__(self, idx):

        if self.is_labeled:

            return self.sen_a[idx], self.sen_b[idx], self.label[idx]

        else:

            return self.sen_a[idx], self.sen_b[idx]



class AttnEncoderRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, dropout_p=0.1, max_length=20):

        super(AttnEncoderRNN, self).__init__()

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.dropout_p = dropout_p

        self.max_length = max_length



        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,

                          dropout=0.1, batch_first=True)

        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,

                          dropout=0.1, batch_first=True)

        

    def forward(self, inp):

        embedded = self.embedding(inp)

        embedded = self.dropout(embedded)

        h_0 = self.initHidden(batch_size=embedded.size(0))

        

        output, hidden = self.gru(embedded, h_0)

        

        attn_weights = F.softmax(

            self.attn(torch.cat((output, output), 2)), dim=2)

        attn_applied = torch.bmm(attn_weights, output)



        output = torch.cat((output, attn_applied), 2)

        output = self.attn_combine(output)

        

        output = F.gelu(output)

        output, hidden = self.gru2(output, hidden)

        

        return output[:, -1 ,:]

    

    def initHidden(self, batch_size=1):

        weight = next(self.parameters()).data

        return weight.new(1, batch_size, self.hidden_size).zero_()

        

class ModelForSTS(nn.Module):

    def __init__(self, vocab_size, hidden_size, emb_size, dropout=0.2, eps=1e-8):

        super().__init__()

        self.rnn = AttnEncoderRNN(vocab_size, hidden_size)

        self.emb_layer = nn.Linear(hidden_size, emb_size)

        self.layer_norm = nn.LayerNorm(emb_size, eps=eps)

        self.cossim = nn.CosineSimilarity()

        self.loss_fct = nn.MSELoss()



    def forward(self, sen_a, sen_b, label=None):



        sen_a = self.rnn(sen_a)

        sen_b = self.rnn(sen_b)



        sen_a = self.emb_layer(sen_a)

        sen_a = self.layer_norm(sen_a)

        

        sen_b = self.emb_layer(sen_b)

        sen_b = self.layer_norm(sen_b)

        

        sim = self.cossim(sen_a, sen_b)



        if label is not None:

            loss = self.loss_fct(sim, label)

            return sim, loss



        return sim
train_dataset = STS_Dataset(train_sen_a_bow, train_sen_b_bow, train_df['label'].to_numpy())

test_dataset = STS_Dataset(test_sen_a_bow, test_sen_b_bow)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
model = ModelForSTS(config.vocab_size, config.hidden_size, config.emb_size).cuda()

optimizer = AdamW(model.parameters(), lr=config.lr)

scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=config.warmup,

                                            num_training_steps=config.warmup*10)
max_correct = 0

max_epoch = 0



for epoch in range(config.epoch):

    model.train()

    train_loss = 0

    train_correct = 0.0

    for sen_a, sen_b, label in train_dl:

        sim, loss = model(sen_a.cuda(), sen_b.cuda(), label.cuda())

        loss.backward()

        optimizer.step()

        scheduler.step()

        model.zero_grad()

        train_loss += loss.item()

        for s, l in zip(sim, label):

            if l+s > 1.0 + config.threshold or l+s < config.threshold:

                train_correct += 1.0

    print("[EPOCH : " + str(epoch+1) +"]")

    print("train_loss : " + str(train_loss))

    print("train_acc : " + str(train_correct/len(train_dataset)))

    

    model.eval()

    test_oup = {'id': [], 'label': []}

    test_id = 40001

    for sen_a, sen_b in test_dl:

        batch_sim = model(sen_a.cuda(), sen_b.cuda())

        batch_sim = batch_sim.tolist()

        for sim in batch_sim:

            test_oup['id'].append(test_id)

            test_id += 1

            if sim > config.threshold:

                test_oup['label'].append(1)

            else:

                test_oup['label'].append(0)

    oup_file_name = "epoch_"+ str(epoch+1) + ".csv"

    pd.DataFrame(test_oup).to_csv(oup_file_name, index=False)