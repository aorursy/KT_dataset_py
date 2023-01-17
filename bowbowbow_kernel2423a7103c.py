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
import argparse

import pickle

from datetime import datetime

import socket

import csv

import math



import torch

import torch.utils.data

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from tensorboardX import SummaryWriter

from sklearn import metrics

from tqdm import tqdm
class Hyperparameters:

    batch_size = 32

    lr = 2e-4

    n_epochs = 30

    train_data_path = "/kaggle/input/lydzc4rn3f67w7z/train.csv"

    test_data_path = "/kaggle/input/lydzc4rn3f67w7z/test.csv"

    model_name = "ESIM_ensemble5_batch32_0.95_CosineAnnealingLR"

    glove_path = "/kaggle/input/my-glove300/glove300.pkl"

    report_acc = 200

    max_norm = 1.0

    patience = 10

    hidden_size = 300

    weight_rate = 0.75

    split_ratio = 0.95

    train_glove = False

    



hp = Hyperparameters()
def split_train_and_valid(data, split_ratio=0.9):

    split_index = int(len(data) * split_ratio)

    np.random.seed(10)

    items = np.array(data)

    shuffle_indices = np.random.permutation(np.arange(len(items)))

    items = list(items[shuffle_indices])

    train_set = items[:split_index]

    valid_set = items[split_index:]



    print('total train_set:', len(train_set))

    print('total valid_set:', len(valid_set))



    return train_set, valid_set





def load_csv(path):

    data = []

    with open(path, 'r') as f:

        reader = csv.reader(f)

        for i, row in enumerate(reader):

            if i == 0: continue

            data.append(row)

    return data



def make_logdir(model_name):

    current_time = datetime.now().strftime('%m.%d_%H.%M.%S')

    logdir = os.path.join('runs', current_time + '_' + socket.gethostname() + '_' + model_name)

    return logdir



def write_submission(test_data, Y_hat, path):

    with open(path, 'w') as f:

        f.write("id,label\n")

        for i, item in enumerate(test_data):

            f.write("{},{}\n".format(item[0], Y_hat[i]))
class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, hp, glove):

        self.hp = hp

        self.glove = glove



        self.sents1, self.sents2, self.labels = [], [], []



        for item in data:

            self.sents1.append(list(map(lambda x: glove['dic'][x] + 1, item[1].split())))

            self.sents2.append(list(map(lambda x: glove['dic'][x] + 1, item[2].split())))

            if len(item) == 3:

                self.labels.append(0)

            else:

                self.labels.append(int(item[3]))



    def __len__(self):

        return len(self.labels)



    def __getitem__(self, idx):

        sent1, sent2, label = self.sents1[idx], self.sents2[idx], self.labels[idx]

        return sent1, sent2, label





def pad(batch):

    max_seq_len = 0

    sents1, sents2, labels, sents1_mask, sents2_mask = [], [], [], [], []

    for item in batch:

        sents1.append(item[0])

        sents2.append(item[1])

        labels.append(item[2])



        sents1_mask.append([1] * len(sents1[-1]))

        sents2_mask.append([1] * len(sents2[-1]))



        max_seq_len = max(max_seq_len, len(sents1[-1]))

        max_seq_len = max(max_seq_len, len(sents2[-1]))



    #  pads to the longest sample

    for i in range(len(sents1)):

        sents1[i] = sents1[i] + [0] * (max_seq_len - len(sents1[i]))

        sents2[i] = sents2[i] + [0] * (max_seq_len - len(sents2[i]))



        sents1_mask[i] = sents1_mask[i] + [0] * (max_seq_len - len(sents1_mask[i]))

        sents2_mask[i] = sents2_mask[i] + [0] * (max_seq_len - len(sents2_mask[i]))



    sents1 = torch.LongTensor(sents1)

    sents2 = torch.LongTensor(sents2)

    labels = torch.LongTensor(labels)

    sents1_mask = torch.LongTensor(sents1_mask)

    sents2_mask = torch.LongTensor(sents2_mask)



    return sents1, sents2, labels, sents1_mask, sents2_mask





class LossWeight:

    def __init__(self, labels, hp):

        self.labels = labels

        self.hp = hp



    def get_loss_weights(self):

        label_cnt = {0: 0, 1: 0}

        for label in self.labels:

            label_cnt[label] += 1



        min_cnt = min([cnt for idx, cnt in label_cnt.items()])



        loss_weights = [0] * 2

        for label in [0, 1]:

            if label_cnt[label] == 0:

                loss_weights[label] = 0

            else:

                loss_weights[label] = math.pow(min_cnt / label_cnt[label], self.hp.weight_rate)



        print('loss_weights :', loss_weights)

        return torch.FloatTensor(loss_weights).cuda()

# https://github.com/maciejkula/glove-python

# !pip install glove_python 
if hp.train_glove:

    from glove import Corpus, Glove



    train_data = load_csv(hp.train_data_path)

    test_data = load_csv(hp.test_data_path)

    data = train_data + test_data



    words = []

    lines = []

    for item in data:

        lines.append(item[1].split())

        lines.append(item[2].split())



        words.extend(item[1].split())

        words.extend(item[2].split())



    print('total words:', len(words))

    words = sorted(list(set(words)), key=lambda x: int(x))

    print('total unique words:', len(words))



    # Creating a corpus object

    corpus = Corpus()



    # Training the corpus to generate the co occurence matrix which is used in GloVe

    corpus.fit(lines, window=10)



    glove = Glove(no_components=hp.hidden_size)

    glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=True)

    glove.add_dictionary(corpus.dictionary)



    word2vec = {}

    for word in words:

        word2vec[word] = glove.word_vectors[glove.dictionary[word]]



    print('len(glove.word_vectors):', len(glove.word_vectors))



    with open(hp.glove_path, "wb") as f:

        pickle.dump({

            'dic': glove.dictionary,

            'vectors': glove.word_vectors,

        }, f)


class ESIM(nn.Module):

    def __init__(self, num_classes, hidden_size):

        super().__init__()

        # Reused from https://github.com/pengshuang/Text-Similarity/blob/master/models/ESIM.py

        self.dropout = 0.5

        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        self.lstm2 = nn.LSTM(hidden_size * 8, hidden_size, batch_first=True, bidirectional=True)



        self.fc = nn.Sequential(

            nn.BatchNorm1d(hidden_size * 8),

            nn.Linear(hidden_size * 8, hidden_size),

            nn.ELU(inplace=True),

            nn.BatchNorm1d(hidden_size),

            nn.Dropout(self.dropout),

            nn.Linear(hidden_size, hidden_size),

            nn.ELU(inplace=True),

            nn.BatchNorm1d(hidden_size),

            nn.Dropout(self.dropout),

            nn.Linear(hidden_size, num_classes),

            nn.Softmax(dim=-1)

        )



    def soft_attention_align(self, x1, x2, mask1, mask2):

        '''

        x1: batch_size * seq_len * dim

        x2: batch_size * seq_len * dim

        '''

        # attention: batch_size * seq_len * seq_len

        attention = torch.matmul(x1, x2.transpose(1, 2))

        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))

        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))



        # weight: batch_size * seq_len * seq_len

        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)

        x1_align = torch.matmul(weight1, x2)

        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)

        x2_align = torch.matmul(weight2, x1)

        # x_align: batch_size * seq_len * hidden_size



        return x1_align, x2_align



    def submul(self, x1, x2):

        mul = x1 * x2

        sub = x1 - x2

        return torch.cat([sub, mul], -1)



    def apply_multiple(self, x):

        # input: batch_size * seq_len * (2 * hidden_size)

        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)

        # output: batch_size * (4 * hidden_size)

        return torch.cat([p1, p2], 1)



    def forward(self, x1, x2, mask1, mask2):

        o1, _ = self.lstm1(x1)

        o2, _ = self.lstm1(x2)



        # Attention

        # batch_size * seq_len * hidden_size

        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)



        # Compose

        # batch_size * seq_len * (8 * hidden_size)

        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)

        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)



        # batch_size * seq_len * (2 * hidden_size)

        q1_compose, _ = self.lstm2(q1_combined)

        q2_compose, _ = self.lstm2(q2_combined)



        # Aggregate

        # input: batch_size * seq_len * (2 * hidden_size)

        # output: batch_size * (4 * hidden_size)

        q1_rep = self.apply_multiple(q1_compose)

        q2_rep = self.apply_multiple(q2_compose)



        # Classifier

        x = torch.cat([q1_rep, q2_rep], -1)

        similarity = self.fc(x)

        return similarity





class Model(nn.Module):

    def __init__(self, num_classes, hidden_size, vocab_size=10000, glove=None):

        super().__init__()

        if glove:

            pretrain = torch.from_numpy(glove['vectors']).float().cuda()

            self.embed = nn.Embedding(hidden_size, hidden_size).from_pretrained(pretrain)

        else:

            self.embed = nn.Embedding(vocab_size, hidden_size)



        self.bn_embed = nn.BatchNorm1d(hidden_size)



        self.esim1 = ESIM(num_classes, hidden_size)

        self.esim2 = ESIM(num_classes, hidden_size)

        self.esim3 = ESIM(num_classes, hidden_size)

        self.esim4 = ESIM(num_classes, hidden_size)

        self.esim5 = ESIM(num_classes, hidden_size)



    def forward(self, sents1, sents2, labels, sents1_mask, sents2_mask):

        x1 = self.bn_embed(self.embed(sents1).transpose(1, 2).contiguous()).transpose(1, 2)

        x2 = self.bn_embed(self.embed(sents2).transpose(1, 2).contiguous()).transpose(1, 2)

        mask1, mask2 = sents1.eq(0), sents2.eq(0)



        sim1 = self.esim1(x1, x2, mask1, mask2)

        sim2 = self.esim2(x1, x2, mask1, mask2)

        sim3 = self.esim3(x1, x2, mask1, mask2)

        sim4 = self.esim4(x1, x2, mask1, mask2)

        sim5 = self.esim5(x1, x2, mask1, mask2)



        sim = (sim1 + sim2 + sim3 + sim4 + sim5) / 5.0



        return sim
def train(epoch, model, iterator, optimizer, criterion):

    model.train()

    total_loss, n_batch = 0, len(iterator)

    tmp_loss = 0

    pbar = tqdm(total=n_batch)

    tmp_labels, tmp_y = [], []

    for step, batch in enumerate(iterator):

        pbar.update(1)

        optimizer.zero_grad()



        sents1, sents2, labels, sents1_mask, sents2_mask = batch



        sents1 = sents1.cuda()

        sents2 = sents2.cuda()

        labels = labels.cuda()

        sents1_mask = sents1_mask.cuda()

        sents2_mask = sents2_mask.cuda()



        logits = model(sents1=sents1,

                       sents2=sents2,

                       labels=labels,

                       sents1_mask=sents1_mask,

                       sents2_mask=sents2_mask)



        loss = criterion(logits, labels)



        total_loss += loss.item()

        tmp_loss += loss.item()



        loss.backward()



        nn.utils.clip_grad_norm_(model.parameters(), hp.max_norm)



        true_labels = labels.cpu().numpy().tolist()

        pred_labels = logits.argmax(-1).cpu().numpy().tolist()



        for k in range(len(true_labels)):

            tmp_labels.append(true_labels[k])

            tmp_y.append(pred_labels[k])



        if step == 0:

            print("=====sanity check======")

            print("sents1[0]:", sents1.cpu().numpy()[0])

            print("sents2[0]:", sents2.cpu().numpy()[0])

            print("true_labels:", true_labels)

            print('pred_labels: ', pred_labels)



        optimizer.step()

        scheduler.step()



        if step > 0 and step % hp.report_acc == 0:

            print("step: {}, loss: {}".format(step, loss.item()))



    print(metrics.classification_report(tmp_labels, tmp_y, digits=4))

    acc = metrics.accuracy_score(tmp_labels, tmp_y)

    writer.add_scalar('train/acc', acc, epoch)

    writer.add_scalar('train/loss_avg', (tmp_loss / hp.report_acc), epoch)



    loss_avg = total_loss / n_batch

    pbar.close()

    return loss_avg





def eval(epoch, model, iterator, criterion, submission=False):

    model.eval()



    Y, Y_hat = [], []

    total_loss, n_batch = 0, len(iterator)

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            sents1, sents2, labels, sents1_mask, sents2_mask = batch



            sents1 = sents1.cuda()

            sents2 = sents2.cuda()

            labels = labels.cuda()

            sents1_mask = sents1_mask.cuda()

            sents2_mask = sents2_mask.cuda()



            logits = model(sents1=sents1,

                           sents2=sents2,

                           labels=labels,

                           sents1_mask=sents1_mask,

                           sents2_mask=sents2_mask)



            loss = criterion(logits, labels)

            total_loss += loss.item()



            true_labels = labels.cpu().numpy().tolist()

            pred_labels = logits.argmax(-1).cpu().numpy().tolist()



            for k in range(len(true_labels)):

                Y.append(true_labels[k])

                Y_hat.append(pred_labels[k])



    loss_avg = total_loss / n_batch

    acc = metrics.accuracy_score(Y, Y_hat)

    if not submission:

        print('acc:', acc)

        writer.add_scalar('dev/acc', acc, epoch)

        writer.add_scalar('dev/loss_avg', loss_avg, epoch)

        cls_report = metrics.classification_report(Y, Y_hat, digits=4, output_dict=True)

        print(metrics.classification_report(Y, Y_hat, digits=4))



    return acc, Y_hat



logdir = make_logdir(hp.model_name)

writer = SummaryWriter(logdir)



train_data = load_csv(hp.train_data_path)

test_data = load_csv(hp.test_data_path)



train_data, valid_data = split_train_and_valid(train_data, split_ratio=hp.split_ratio)



with open(hp.glove_path, "rb") as f:

    glove = pickle.load(f)

    glove['vectors'] = np.array([[0.0] * hp.hidden_size] + glove['vectors'].tolist())



train_dataset = Dataset(train_data, hp=hp, glove=glove)

valid_dataset = Dataset(valid_data, hp=hp, glove=glove)

test_dataset = Dataset(test_data, hp=hp, glove=glove)



train_iter = torch.utils.data.DataLoader(dataset=train_dataset,

                                         batch_size=hp.batch_size,

                                         shuffle=True,

                                         num_workers=8,

                                         pin_memory=True,

                                         collate_fn=pad)



valid_iter = torch.utils.data.DataLoader(dataset=valid_dataset,

                                         batch_size=hp.batch_size,

                                         shuffle=False,

                                         num_workers=8,

                                         pin_memory=True,

                                         collate_fn=pad)



test_iter = torch.utils.data.DataLoader(dataset=test_dataset,

                                        batch_size=hp.batch_size,

                                        shuffle=False,

                                        num_workers=8,

                                        pin_memory=True,

                                        collate_fn=pad)



model = Model(

    num_classes=2,

    hidden_size=hp.hidden_size,

    vocab_size=len(glove['dic']) + 1,

    glove=glove,

)

model = model.cuda()

model = nn.DataParallel(model)



optimizer = optim.Adam(model.parameters(), lr=hp.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data) // hp.batch_size)

loss_weight = LossWeight(hp=hp, labels=train_dataset.labels).get_loss_weights()

criterion = nn.CrossEntropyLoss(weight=loss_weight)



eval(0, model, valid_iter, criterion)

best_valid_acc = 0.0

over_fitting = 0

for epoch in range(1, hp.n_epochs + 1):

    loss_avg = train(epoch, model, train_iter, optimizer, criterion)



    print("=========eval at epoch={}=========".format(epoch))

    valid_acc, _ = eval(epoch, model, valid_iter, criterion)

    if best_valid_acc < valid_acc:

        over_fitting = 0

        print('best_valid_acc update! valid_acc: {}'.format(valid_acc))

        best_valid_acc = valid_acc

        # model_path = os.path.join(logdir, 'epoch_{:02d}__acc_{:.3f}'.format(epoch, valid_acc))

        # torch.save(model.state_dict(), "{}.pt".format(model_path))

        # print("weights were saved to {}.pt".format(model_path))



        _, test_Y_hat = eval(epoch, model, test_iter, criterion, submission=True)

        submission_path = os.path.join(logdir, 'epoch_{:02d}__acc_{:.3f}__submission.csv'.format(epoch, valid_acc))

        write_submission(test_data, test_Y_hat, submission_path)

        print("submission were saved to {}".format(submission_path))

    else:

        over_fitting += 1

        if over_fitting >= hp.patience:

            break