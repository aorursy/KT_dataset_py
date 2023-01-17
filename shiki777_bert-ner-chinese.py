import sys

package_dir = "../input/pytorch-pre"

sys.path.append(package_dir)

from sklearn.metrics import f1_score, classification_report

import numpy as np

from pytorch_pretrained import BertModel, BertTokenizer

import torch.optim as optim

import os

import operator

import time

import torch

from functools import reduce

from torch import nn

from tqdm import tqdm

import random

from torch.utils.data import TensorDataset, DataLoader

import warnings

import numpy as np

warnings.filterwarnings("ignore")
class Config(object):

    """配置参数"""

    def __init__(self):

        self.data_path = '../input/datas-1/source_BIO_2014_cropus.txt'   # 文本验证集

        self.label_path = '../input/datas-1/target_BIO_2014_cropus.txt' # 标签验证集

        self.bert_path = '../input/bert-model'

        self.label_dic = {"B_PER":0, "I_PER":1, "B_T":2, "I_T":3, "B_ORG":4, "I_ORG":5, "B_LOC":6, "I_LOC":7, "O":8}

        self.tagset_size = len(self.label_dic)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_epochs = 1  # epoch数

        self.batch_size = 16

        self.pad_size = 100  # 每句话处理成的长度(短填长切)

        self.learning_rate = 2e-5 # 学习率

        self.learning_rate_decay = 5e-6  # 学习率衰减

        self.hidden_size = 100

        self.embedding_dim = 768

        self.num_layers = 1

        self.dropout = 1
def train_val_split(X,Y,valid_size,random_state=2018, shuffle=True):

    data = []

    for data_x, data_y in tqdm(zip(X, Y), desc='Merge'):

        data.append((data_x, data_y))

    del X, Y

    N = len(data)

    test_size = int(N * valid_size)

    if shuffle:

        random.seed(random_state)

        random.shuffle(data)

    valid = data[:test_size]

    train = data[test_size:]

    return train, valid
def produce_data(config):

    targets,sentences = [],[]

    f1 = open(config.data_path,'r',encoding = 'utf-8')

    f2 = open(config.label_path,'r',encoding = 'utf-8')

    for sent,target in tqdm(zip(f1,f2),desc = 'text_to_id'):

        assert len(sent.strip('\n').split()) == len(target.strip('\n').split())

        sentences.append(sent.strip('\n').split())

        targets.append(target.strip('\n').split())

    sentences = sentences # create a small dataset

    targets = targets #create a small dataset

    train,val = train_val_split(sentences,targets,valid_size = 0.2)

    return train,val
class InputFeatures(object):

    def __init__(self, input_id, label_id, input_mask,output_mask):

        self.input_id = input_id

        self.label_id = label_id

        self.input_mask = input_mask

        self.output_mask = output_mask
def build_dataset(config,x_train,y_train):

    def load_dataset(x_train,y_train,pad_size,label_dic,tokenizer):

        """

        :param data_path:文本数据路径

        :param label_path:标签数据路径

        :param pad_size:每个句子最大长度

        :param label_dic:词性种类表

        :return:

        """

        result=[]

        x_train = x_train[:3000]

        y_train = y_train[:3000]

        for tokens,label in zip(x_train,y_train):

            

            assert len(tokens) == len(label)



            if len(tokens) > pad_size - 2:  # 大于最大长度进行截断

                tokens = tokens[0:(pad_size - 2)]

                label = label[0:(pad_size - 2)]





            tokens_c_s = ['[CLS]'] + tokens + ['[SEP]']

            label_c_s = ' '.join(label)





            input_ids=tokenizer.convert_tokens_to_ids(tokens_c_s) #分词 --> 字典

            label_ids=[label_dic[i] for i in label_c_s.split()]

            label_ids = [9] + label_ids + [10]



            input_mask=[1]*len(input_ids)

            output_mask = [1] * len(label_ids)

            

            if len(input_ids) < pad_size:

                input_ids = input_ids + ([0]*(pad_size-len(input_ids)))

                input_mask += ([0]*(pad_size-len(input_mask)))

            if len(label_ids) < pad_size:

                label_ids = label_ids+([-1]*(pad_size-len(label_ids)))

                output_mask +=([0]*(pad_size-len(output_mask)))

                

                



            assert len(input_ids) == pad_size

            assert len(input_mask) == pad_size

            assert len(label_ids) == pad_size

            assert len(output_mask) == pad_size



            #              处理后数据

            # -------------------------------------------

            # 原始:           我 是 中 国 人

            # 分词:     [CLS] 我 是 中 国 人 [SEP]

            # input_id:  101  2 12 13 16 14  102  0  0  0

            # input_mask:  1  1  1  1  1  1    1  0  0  0

            # label_id:       T  T  0  0  0   -1 -1 -1 -1 -1

            # output_mask: 0  1  1  1  1  1    0  0  0  0

            feature = InputFeatures(input_id=input_ids, label_id=label_ids, input_mask=input_mask,

                                    output_mask=output_mask)

            result.append(feature)

        return result





    data = load_dataset(x_train,y_train,config.pad_size,config.label_dic,config.tokenizer)

    ids = torch.LongTensor([_.input_id for _ in data])

    input_masks = torch.LongTensor([_.input_mask for _ in data])

    tags = torch.LongTensor([_.label_id for _ in data])

    output_masks = torch.LongTensor([_.output_mask for _ in data])

    dataset = TensorDataset(ids, input_masks, tags, output_masks)

    return dataset
def built_train_dataset(config,train_datas):

    x_train,y_train = zip(*train_datas)

    train_dataset=build_dataset(config,x_train,y_train)

    return DataLoader(train_dataset,shuffle=False,batch_size=config.batch_size,drop_last=True)





def built_dev_dataset(config,dev_datas):

    x_val,y_val = zip(*dev_datas)

    dev_dataset=build_dataset(config,x_val,y_val)

    return DataLoader(dev_dataset,shuffle=False,batch_size=config.batch_size,drop_last=True)
START_TAG = "CLS"

STOP_TAG = "SEP"

class Bert_LSTMCRF(nn.Module):

    def __init__(self,config):

        super(Bert_LSTMCRF, self).__init__()

        self.tag2id = {"B_PER":0, "I_PER":1, "B_T":2, "I_T":3, "B_ORG":4, "I_ORG":5, "B_LOC":6, "I_LOC":7, "O":8,START_TAG :9,STOP_TAG:10}

        self.tag2id_size = len(self.tag2id)

        self.bert_path = config.bert_path

        self.bert = BertModel.from_pretrained(self.bert_path)

        for p in self.bert.parameters():

          p.requires_grad = False

        self.batch_size = config.batch_size

        self.embedding_dim = 768

        self.hidden_dim = config.hidden_size

        # 概率转移矩阵

        self.transitions = nn.Parameter(torch.randn(self.tag2id_size, self.tag2id_size))

        self.transitions.data[:, self.tag2id[START_TAG]] = -1000.

        self.transitions.data[self.tag2id[STOP_TAG], :] = -1000.



        # batch_first=True：batch_size在第一维而非第二维

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,

                            num_layers = 1, bidirectional = True,

                            batch_first = True,dropout=config.dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag2id_size)



    # 每帧对应的隐向量

    # input =[contex,mask]

    def get_lstm_features(self, batch_sentence,input_mask):

        

        with torch.no_grad():

            embeddings, _ = self.bert(batch_sentence, attention_mask=input_mask, output_all_encoded_layers=False)

        hidden = (torch.randn(2, self.batch_size, self.hidden_dim // 2).to(config.device),

                  torch.randn(2, self.batch_size, self.hidden_dim // 2).to(config.device))



        lstm_out, _hidden = self.lstm(embeddings, hidden)

        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)

        # 降维到标签空间

        

        return self.hidden2tag(lstm_out)



    # 真实路径的分值（针对一个实例而非一个batch）

    def real_path_score(self, logits, tags):

        score = torch.zeros(1).to(config.device)

        tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype = torch.long).to(config.device), tags])

        # 累加每帧的转移和发射

        for i, logit in enumerate(logits):

            # len(tags) = len(logits) + 1

            transition_score = self.transitions[tags[i], tags[i + 1]]

            emission_score = logit[tags[i + 1]]

            score += transition_score + emission_score

        # 处理结尾

        score += self.transitions[tags[-1], self.tag2id[STOP_TAG]]

        return score



    # 数值稳定性

    def log_sum_exp(self, smat):

        vmax = smat.max(dim = 0, keepdim = True).values    # 每列的最大值

        return (smat - vmax).exp().sum(axis = 0, keepdim = True).log() + vmax



    # 概率归一化分母（针对一个实例而非一个batch）

    def total_score(self, logits):

        alpha = torch.full((1, self.tag2id_size), -1000.).to(config.device)

        alpha[0][self.tag2id[START_TAG]] = 0

        # 沿时间轴dp

        for logit in logits:

            alpha = self.log_sum_exp(alpha.T + logit.unsqueeze(0) + self.transitions)

        # STOP，发射分值0，转移分值为列向量（self.tag2id[STOP_TAG]外加上[]）

        return self.log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag2id[STOP_TAG]]]).flatten()



    # 负对数似然

    def neg_log_likelihood(self, batch_sentences, batch_tags, input_mask):



        batch_length = torch.sum(input_mask,axis = 1)

        batch_logits = self.get_lstm_features(batch_sentences,input_mask)

        real_path_score = torch.zeros(1).to(config.device)

        total_score = torch.zeros(1).to(config.device)

        # 一个batch求和

        for logits, tags, len in zip(batch_logits, batch_tags, batch_length):

            # mask



            logits = logits[:len]

            tags = tags[:len]

            real_path_score += self.real_path_score(logits, tags)

            total_score += self.total_score(logits)



        return total_score - real_path_score



    # 维特比解码

    def viterbi_decode(self, logits):

        backtrace = []

        # 初始化

        alpha = torch.full((1, len(self.tag2id)), -1000.).to(config.device)

        alpha[0][self.tag2id[START_TAG]] = 0

        # 沿时间轴dp

        for frame in logits:

            smat = alpha.T + frame.unsqueeze(0) + self.transitions

            backtrace.append(smat.argmax(0))    # 当前时刻，每个状态的最优来源

            alpha = self.log_sum_exp(smat)

        smat = alpha.T + 0 + self.transitions[:, [self.tag2id[STOP_TAG]]]

        # 回溯路径

        best_tag_id = smat.flatten().argmax().item()

        best_path = [best_tag_id]

        # 从[1:]开始，去掉START_TAG

        for bptrs_t in reversed(backtrace[1:]):

            best_tag_id = bptrs_t[best_tag_id].item()

            best_path.append(best_tag_id)

        # 最优路径分值和最优路径

        return self.log_sum_exp(smat).item(), best_path[::-1]



    # 推断

    def forward(self, batch_sentences,input_mask):

        

        batch_sentences = torch.tensor(batch_sentences, dtype = torch.long).to(config.device)

        batch_length = torch.sum(torch.tensor(input_mask),axis = 1).to(config.device)

        

        batch_logits = self.get_lstm_features(batch_sentences,input_mask)



        batch_scores = []

        batch_paths = []

        # 计算一个batch

        for logits, len in zip(batch_logits, batch_length):

            logits = logits[:len]

            score, path = self.viterbi_decode(logits)

            batch_scores.append(score)

            batch_paths.append(path)

        # batch_scores = batch_scores.cpu()

        # batch_paths = batch_paths.cpu()

        return batch_scores, batch_paths
def eval(model,dataset):

#     f1 = 0

    model.eval()

    with torch.no_grad():

      

      y_predicts = []

      y_labels = []

      for i,batch in enumerate(dataset):

          input_id, input_mask, label, output_mask = batch

          input_id = input_id.to(config.device)

          input_mask = input_mask.to(config.device)

          label = label.to(config.device)

          _,path = model(input_id,input_mask)

          tmp = reduce(operator.add, path)

          y_predicts += tmp

          label = label.view(1,-1)

          label = label[label != -1]

          # print(label)

          y_labels.append(label)

      y_true = torch.cat(y_labels,dim=0)

      y_pred = np.array(y_predicts)

      y_true = y_true.cpu().numpy()

      y_pred = y_pred[y_true!=8].astype(np.int)

      y_true = y_true[y_true!=8].astype(np.int)

      return f1_score(y_true,y_pred,average="macro")

config = Config()

train_datas,dev_datas = produce_data(config)

train_dataset = built_train_dataset(config,train_datas)

dev_dataset = built_dev_dataset(config,dev_datas)
model = Bert_LSTMCRF(config)

model = model.to(config.device)

if os.path.exists("/kaggle/working/params.pkl"):

    model.load_state_dict(torch.load("/kaggle/woking/params.pkl"))

    print("model load success!")
optimizer = optim.SGD(model.parameters(),lr = config.learning_rate,momentum=0.9)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
best_score = 0

for epoch in range(config.num_epochs):

    model.train()

    total_loss = 0

    batch_count = 0

    start_time = time.time()

    for i,batch in enumerate(train_dataset):

        model.zero_grad()

        input_id, input_mask, label, output_mask = batch

        input_id = input_id.to(config.device)

        input_mask = input_mask.to(config.device)

        label = label.to(config.device)

        loss = model.neg_log_likelihood(input_id,label,input_mask)

        total_loss += loss.tolist()[0]

        

        batch_count += 1

        loss.backward()

        optimizer.step()

    f1_score1 = eval(model,dev_dataset)

    if f1_score1 > best_score:

      torch.save(model.state_dict(), "/kaggle/working/params.pkl")

    print("epoch: {}\tloss: {:.2f}\ttime: {:.1f} sec\tf1_score: {:.4f}".format(epoch + 1, total_loss / batch_count,time.time() - start_time,f1_score1))
print(os.path.exists('/kaggle/working'))