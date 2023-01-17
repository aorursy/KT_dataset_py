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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec

import torch

from torch import nn

from transformers import AdamW, get_linear_schedule_with_warmup

import os

from hbconfig import Config
train_data = "/kaggle/input/g8wamyvdmxl7qyr/1.train.csv"

test_data = "/kaggle/input/g8wamyvdmxl7qyr/1.test.csv"



train_df = pd.read_csv(train_data)

test_df = pd.read_csv(test_data)



train_df.head()
# os.environ["WANDB_API_KEY"] = ""



# wandb.init(project="STS-FNN")

# config = wandb.config

# # config.model = "BOW"

# # config.model = "TF-IDF"

# config.model = "word2vec"

# if config.model == "word2vec":

#     config.vocab_size = 2048

# else:

#     config.vocab_size = 8210

# config.hidden_layer_size = [config.vocab_size, 1024, 512]

# config.emb_size = 128

# config.lr = 5e-5

# config.batch_size = 128

# config.warmup = 1000

# config.valid_data_rate = 0.08

# config.epoch = 50

# config.threshold = 0.5
config = Config

# config.model = "BOW"

# config.model = "TF-IDF"

config.model = "word2vec"

if config.model == "word2vec":

    config.vocab_size = 2048

else:

    config.vocab_size = 8210

config.hidden_layer_size = [config.vocab_size, 1024, 512]

config.emb_size = 128

config.lr = 5e-5

config.batch_size = 128

config.warmup = 1000

config.valid_data_rate = 0.08

config.epoch = 50

config.threshold = 0.5
if config.model == "word2vec":

    w2c_corpus = []

    for s in train_df['sentence1'].append(train_df['sentence2']):

        w2c_corpus.append(s.split())

    for s in test_df['sentence1'].append(test_df['sentence2']):

        w2c_corpus.append(s.split())

    

    w2v = Word2Vec(w2c_corpus, size=config.vocab_size, window = 2, min_count=1, workers=4, iter=100, sg=1)



    train_sen_a_bow = np.array([sum(w2v.wv[token] for token in s.split())/len(s.split()) for s in train_df['sentence1']])

    train_sen_b_bow = np.array([sum(w2v.wv[token] for token in s.split())/len(s.split()) for s in train_df['sentence2']])

    test_sen_a_bow = np.array([sum(w2v.wv[token] for token in s.split())/len(s.split()) for s in test_df['sentence1']])

    test_sen_b_bow = np.array([sum(w2v.wv[token] for token in s.split())/len(s.split()) for s in test_df['sentence2']])

else:

    if config.model == "BOW":

        vector = CountVectorizer()

    elif config.model == "TF-IDF":

        vector = TfidfVectorizer()

        

    vector.fit(train_df['sentence1'].append(train_df['sentence2']))



    train_sen_a_bow = vector.transform(train_df['sentence1']).toarray()

    train_sen_b_bow = vector.transform(train_df['sentence2']).toarray()

    test_sen_a_bow = vector.transform(test_df['sentence1']).toarray()

    test_sen_b_bow = vector.transform(test_df['sentence2']).toarray()
class STS_Dataset(torch.utils.data.Dataset):

    def __init__(self, sen_a, sen_b, label=None):

        self.sen_a = torch.tensor(sen_a, dtype=torch.float32)

        self.sen_b = torch.tensor(sen_b, dtype=torch.float32)

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



class LinearLayer(nn.Module):

    def __init__(self, inp, oup, dropout=0.2, eps=1e-8):

        super().__init__()

        self.layer = nn.Linear(inp, oup)

        self.act_fn = nn.GELU()

        self.layer_norm = nn.LayerNorm(oup, eps=eps)

        self.dropout = nn.Dropout(dropout)

    

    def forward(self, inp):

        oup = self.layer(inp)

        oup = self.act_fn(oup)

        oup = self.layer_norm(oup)

        oup = self.dropout(oup)

        

        return oup

        

class ModelForSTS(nn.Module):

    def __init__(self, hidden_layer_size, emb_size, dropout=0.2, eps=1e-8):

        super().__init__()

        self.linear = nn.ModuleList([LinearLayer(inp, oup) for inp, oup in zip(hidden_layer_size, hidden_layer_size[1:])])

        self.emb_layer = nn.Linear(hidden_layer_size[-1], emb_size)

        self.layer_norm = nn.LayerNorm(emb_size, eps=eps)

        self.cossim = nn.CosineSimilarity()

        self.loss_fct = nn.MSELoss()



    def forward(self, sen_a, sen_b, label=None):



        for layer in self.linear:

            sen_a = layer(sen_a)

            sen_b = layer(sen_b)



        sen_a = self.emb_layer(sen_a)

        sen_a = self.layer_norm(sen_a)

        

        sen_b = self.emb_layer(sen_b)

        sen_b = self.layer_norm(sen_b)

        

        sim = self.cossim(sen_a, sen_b)

        

        if label is not None:

            loss = self.loss_fct(sim, label)

            return sim, loss



        return sim
# mask = np.random.choice(a=[False, True], size=40000, p=[config.valid_data_rate, 1.0-config.valid_data_rate])

# train_dataset = STS_Dataset(train_sen_a_bow[mask], train_sen_b_bow[mask], train_df['label'][mask].to_numpy())

# valid_dataset = STS_Dataset(train_sen_a_bow[~mask], train_sen_b_bow[~mask], train_df['label'][~mask].to_numpy())

# test_dataset = STS_Dataset(test_sen_a_bow, test_sen_b_bow)

# train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

# test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
train_dataset = STS_Dataset(train_sen_a_bow, train_sen_b_bow, train_df['label'].to_numpy())

test_dataset = STS_Dataset(test_sen_a_bow, test_sen_b_bow)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
model = ModelForSTS(config.hidden_layer_size, config.emb_size).cuda()

optimizer = AdamW(model.parameters(), lr=config.lr)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup, num_training_steps=config.warmup*10)
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

#     valid_loss = 0

#     valid_correct = 0.0

#     for sen_a, sen_b, label in valid_dl:

#         sim, loss = model(sen_a.cuda(), sen_b.cuda(), label.cuda())

#         valid_loss += loss.item()

#         for s, l in zip(sim, label):

#             if l+s > 1.0 + config.threshold or l+s < config.threshold:

#                 valid_correct += 1.0

#     print("valid_loss : " + str(valid_loss))

#     print("valid_acc : " + str(valid_correct/len(valid_dataset)))

    

#     if valid_correct > max_correct:

#         max_correct = valid_correct

#         max_epoch = epoch

        

    

#     wandb.log({"train_acc": train_correct/len(train_dataset),

#                "valid_acc": valid_correct/len(valid_dataset),

#                "train_loss": train_loss,

#                "valid_loss": valid_loss})

    

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

    

print(max_epoch)