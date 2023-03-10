# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import torch
# Any results you write to the current directory are saved as output.
torch.cuda.is_available()
import numpy as np # linear algebra
import pandas as pd
import os
import torch.utils.data as data
import torch
import pickle
from os import listdir
from os.path import join
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

base_epoch = 2
start_article = 360000
base_step = 360000 / 15
end_article = start_article + 15 * 12000
end_article
def pre_read(fold_dir):
        args = {}
        f = open(fold_dir + 'train_article_embedding_data.pickle','rb')
        args['train_article_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_title_embedding_data.pickle','rb')
        args['train_title_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_label_list_data.pickle','rb')
        args['train_label_list_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_article_embedding_data.pickle','rb')
        args['val_article_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_title_embedding_data.pickle','rb')
        args['val_title_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_label_list_data.pickle','rb')
        args['val_label_list_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'label_vocab.pickle','rb')
        args['label_vocab'] = pickle.load(f)
        f.close()
        args['article_dim'] = len(args['train_article_embedding_data'][0])
        args['title_dim'] = len(args['train_title_embedding_data'][0])
        args['label_dim'] = len(args['label_vocab'])
        args['train_size'] = len(args['train_article_embedding_data'])
        args['embed_dim'] = len(args['train_article_embedding_data'][0][0])
        return args

args = pre_read('../input/train-validation-data-split/')
class DatasetFromFile(data.Dataset):
    def __init__(self, args, input_transform=None, target_transform=True, start_step = 0, end_step = 0):
        super(DatasetFromFile, self).__init__()
        self.train_article_embedding_data = args['train_article_embedding_data'][start_step:end_step]
        self.train_title_embedding_data = args['train_title_embedding_data'][start_step:end_step]
        self.train_label_list_data = args['train_label_list_data'][start_step:end_step]
        self.label_vocab = args['label_vocab']
        self.input_transform = input_transform
        self.target_transform = target_transform

    def one_hot_label_embedding(self, label_list, label_vocab):
        tmp = label_list.copy()
        l = [0] * len(label_vocab)
        for j in tmp:
            if j in label_vocab:
                l[label_vocab[j]] = 1
        return l

    # ???__getitem__????????????????????????????????????transformation???????????????
    # ????????????????????? `input = self.input_transforms(input)`
    # ????????? self.input_transforms???????????????"????????????"???????????????callable???
    # ???????????? "????????????????????????"???????????????????????????????????????????????????
    def __getitem__(self, index):
        article_input = self.train_article_embedding_data[index]
        title_input = self.train_title_embedding_data[index]
        target_tags = self.train_label_list_data[index]
        if self.input_transform:
            input = [torch.FloatTensor(article_input),torch.FloatTensor(title_input)]
        else:
            input = torch.FloatTensor(article_input)
        if self.target_transform:
            target_tags = self.one_hot_label_embedding(target_tags, self.label_vocab)
        
        label = torch.FloatTensor(target_tags)
        return input, label

    def __len__(self):
        return len(self.train_title_embedding_data)

class ValidationFromFile(data.Dataset):
    def __init__(self, args, input_transform=None, target_transform=True):
        super(ValidationFromFile, self).__init__()
        self.val_article_embedding_data = args['val_article_embedding_data']
        self.val_title_embedding_data = args['val_title_embedding_data']
        self.val_label_list_data = args['val_label_list_data']
        self.label_vocab = args['label_vocab']
        self.input_transform = input_transform
        self.target_transform = target_transform

    def one_hot_label_embedding(self, label_list, label_vocab):
        tmp = label_list.copy()
        l = [0] * len(label_vocab)
        for j in tmp:
            if j in label_vocab:
                l[label_vocab[j]] = 1
        return l

    def __getitem__(self, index):
        article_input = self.val_article_embedding_data[index]
        title_input = self.val_title_embedding_data[index]
        target_tags = self.val_label_list_data[index]
        if self.input_transform:
            input = [torch.FloatTensor(article_input),torch.FloatTensor(title_input)]
        else:
            input = torch.FloatTensor(article_input)
        if self.target_transform:
            target_tags = self.one_hot_label_embedding(target_tags, self.label_vocab)
        label = torch.FloatTensor(target_tags)
        return input, label

    def __len__(self):
        return len(self.val_article_embedding_data)
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # we have to reshape Y
        y = y.contiguous().view(n, t, y.size()[1])
        return y

class K_MaxPooling(nn.Module):
    def __init__(self, k):
        super(K_MaxPooling, self).__init__()
        self.k = k
        
    def forward(self, x, dim):
        index = x.topk(self.k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        article_dim = args['article_dim'] #500
        title_dim = args['title_dim'] #15
        label_dim = args['label_dim'] # 36321
        train_size = args['train_size']
        embed_dim = args['embed_dim'] # 300
        # self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)

        D = embed_dim
        C = label_dim
        Ci = 1
        Co = 300#opt['kernel_num']
        #Ks = opt['kernel_sizes']
        Ks1 = [1,2,3,4,5]
        Ks2 = [3,4,5,6,7]
        
        self.tdfc1 = nn.Linear(D, 300)#512)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)
        
        self.tdfc2 = nn.Linear(D, 300)#512)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 290)) for K in Ks1])
        self.convbn1 = nn.ModuleList([nn.BatchNorm2d(Co) for i in range(len(Ks1))])
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 290)) for K in Ks2])
        self.convbn2 = nn.ModuleList([nn.BatchNorm2d(Co) for i in range(len(Ks2))])
        
        self.kmax_pooling = K_MaxPooling(5)
    
        self.fc1 = nn.Linear(33000, C)#4096)
        # self.fc1 = nn.Linear(115200, C)#4096)
        self.bn1 = nn.BatchNorm1d(C)#4096)
        self.fc2 = nn.Linear(C, C)#4096, C)

    def forward(self, x, y):
        batch_size = x.size(0)
        x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1)))
        y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1)))
        x = [F.relu(self.convbn1[i](conv(x))).squeeze(3) for i, conv in enumerate(self.convs1)]
        #print(x)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = [self.kmax_pooling(i, 2).mean(2).squeeze(1) for i in x]
        x = torch.cat(x, 1)
        
        y = [F.relu(self.convbn2[i](conv(y))).squeeze(3) for i, conv in enumerate(self.convs2)]
        # y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]
        y = [self.kmax_pooling(i, 2).mean(2).squeeze(1) for i in y]
        
        y = torch.cat(y, 1)
        x = torch.cat((x, y), 1)
        x = x.view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit
def eval(model, label_dim, val_dataset):
    model.eval()
    predict_label_list, truth_label_list = np.array([0]*label_dim).reshape(1,label_dim), np.array([0]*label_dim).reshape(1,label_dim)
    for step, data in enumerate(val_dataset):
        if step > 50:
            break
        train_x, label = data[0], data[1]
        article_x, title_x = train_x[0], train_x[1]
        article_x, title_x, label = Variable(article_x), Variable(title_x), Variable(label)
        truth_label_list =  np.concatenate((truth_label_list,label.data.numpy()),axis=0)
        batch_size = article_x.size(0)
        article_x, title_x, label = article_x.cuda(), title_x.cuda(), label.cuda()
        output = model(article_x, title_x)
        predict_label_list = np.concatenate((predict_label_list,output.cpu().data.numpy()),axis=0)
    model.train()
    truth_y = truth_label_list[1:]
    predict_y = []
    predict_label_list = predict_label_list[1:]
    for observation in predict_label_list:
        labeled = observation > 0
        sum_labeled = sum(labeled)
        if 0 < sum_labeled <= 5:
            tmp = list(labeled)
        elif sum_labeled > 5:
            tmp = [0]*label_dim
            for i in np.argpartition(observation, -5)[-5:]:
                tmp[i] = 1
        elif sum_labeled == 0:
            tmp = [0]*label_dim
            for i in np.argpartition(observation, -3)[-3:]:
                tmp[i] = 1
        predict_y.append(tmp)
    predict_y = np.array(predict_y)
    F1_score, precision_score, recall_score = [], [], []
    for i in range(len(truth_y)):
        pre = np.array(predict_y[i])
        truth = np.array(truth_y[i])
        r_n, t_n = sum(pre), sum(truth)
        if t_n == 0:
            continue
        tp = (pre == truth)
        for j in range(len(tp)):
            if tp[j] == 1 and truth[j] == 0:
                tp[j] = 0
        tp = sum(tp)
        precision = 1.0 * tp / r_n if r_n != 0 else 0
        recall = 1.0 * tp / t_n
        F1 = 2.0 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        # r(n) ??? t(u) - tp
        # r(n) tp + fp predict ??????
        # t(n) tp + fn truth tags ??????
        F1_score.append(F1)
        precision_score.append(precision)
        recall_score.append(recall)
    F1_score, precision_score, recall_score = np.array(F1_score), np.array(precision_score), np.array(recall_score)
    F1_overall, precision_overall, recall_overall = F1_score.mean(), precision_score.mean(), recall_score.mean()
    return F1_overall, precision_overall, recall_overall, F1_score, precision_score, recall_score
model = torch.load('../input/train-and-train-15/TextCNN_model.pkl',map_location=lambda storage, loc: storage)
tag_dataset = torch.utils.data.DataLoader(
         DatasetFromFile(args, input_transform=True, target_transform=True, start_step = start_article, end_step = end_article), 
         batch_size= 15, shuffle= False)
val_dataset = torch.utils.data.DataLoader(
         ValidationFromFile(args, input_transform=True, target_transform=True), 
         batch_size= 15, shuffle= False)
# model = textCNN(args)
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#????????????
loss_function = nn.MultiLabelSoftMarginLoss()
model.cuda()
from datetime import datetime 
a=datetime.now()
for epoch in range(1):   # ????????????!??????!?????? 3 ???
    for step, data in enumerate(tag_dataset):
        if step % 500 == 0:
            logging.info('Training step {} {} finished!'.format(epoch+base_epoch, step+base_step))
        train_x, label = data[0], data[1]
        article_x, title_x = train_x[0], train_x[1]
        article_x, title_x, label = Variable(article_x), Variable(title_x), Variable(label)
        article_x, title_x, label = article_x.cuda(), title_x.cuda(), label.cuda()
        output = model(article_x, title_x)
        loss = loss_function(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info('Training epoch {} finished!'.format(epoch+base_epoch))
    F1_overall, precision_overall, recall_overall, _, _, _ = eval(model, args['label_dim'], val_dataset)
    print('Epoch[{}] - score: {:.6f} (precision: {:.4f}, recall: {:.4f})'.format( \
                            epoch, F1_overall, precision_overall, recall_overall))
b=datetime.now() 
(b-a).seconds
torch.save(model.cpu(), 'TextCNN_model.pkl')  # ??????????????????
# torch.save(model.state_dict(), 'TextCNN_model_para.pkl')   # ??????????????????????????? (?????????, ????????????)
torch.save(model.cpu().state_dict(), 'TextCNN_model_para.pkl')
# TextCNN_model = torch.load('TextCNN_model.pkl',map_location=lambda storage, loc: storage)
# TextCNN_model_para = textCNN(args)
# TextCNN_model_para.load_state_dict(torch.load('TextCNN_model_para.pkl',map_location=lambda storage, loc: storage))