!pip install transformers==3.1.0
!nvidia-smi
!pip install barbar
import pandas as pd

import numpy as np

import tensorflow as tf

import torch

from torch.nn import BCEWithLogitsLoss, BCELoss

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import pickle

from transformers import *

from tqdm import tqdm, trange

from ast import literal_eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)
# from google.colab import drive 

# drive.mount('/content/drive')
import pandas as pd

import numpy as np
train_data = pd.read_csv('../input/hoc-dataset/train.csv')

dev_data = pd.read_csv('../input/hoc-dataset/dev.csv')

test_data = pd.read_csv('../input/hoc-dataset/test.csv')
dev_data.head(500)
# train_data['1'].value_counts()
# def to_label(train_dataframe):

#     labels = train_dataframe.labels.to_list()

#     train_dataframe.drop(columns=['labels'], inplace=True)

#     ans = np.zeros(shape=(len(labels), 10), dtype=int)

#     for i, s1 in enumerate(labels):

#         l = s1.split(',')

#         for j, s2 in enumerate(l):

#             ans[i, j] = int(s2[-1])

#     header_label = np.arange(10)

#     for header in header_label:

#         train_dataframe[str(header)] = ans[:, header]
# to_label(train_data)

# to_label(dev_data)

# to_label(test_data)
test_data.columns[1:-1]
# BERT:

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 

# XLNet:

xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False) 

# RoBERTa:

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
print(test_data['sentence'][1])

print(len(roberta_tokenizer(test_data['sentence'][1])['input_ids']))

len(roberta_tokenizer.tokenize(test_data['sentence'][1]))

print('Unique sentences -- Train: ', train_data.sentence.nunique() == train_data.shape[0])

print('Null values -- Train: ', train_data.isnull().values.any())

print('Unique sentences -- Dev: ', dev_data.sentence.nunique() == dev_data.shape[0])

print('Null values -- Dev: ', dev_data.isnull().values.any())

print('Unique sentences -- Test: ', test_data.sentence.nunique() == test_data.shape[0])

print('Null values -- Test: ', test_data.isnull().values.any())
print('average sentence length: ', train_data.sentence.str.split().str.len().mean())

print('max sentence length: ', train_data.sentence.str.split().str.len().max())

print('max sentence length: ', dev_data.sentence.str.split().str.len().max())

print('max sentence length: ', test_data.sentence.str.split().str.len().max())
# print(train_data['0'].value_counts())

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True) # tokenizer

# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 

# def remove_long_tokenizer(dataframe, tokenizer):

#     dem = 0

#     i = 0

#     sentences = dataframe['sentence'].tolist()

#     for sentence in tqdm(sentences):

#         input_ids = tokenizer(sentence, padding='max_length', max_length=512, truncation=True)['input_ids']

#         if len(input_ids) > 512:

#             # dataframe.drop(axis=0, index=i, inplace=True)

#             dem += 1

#         i += 1

#     print(f'\nPercentage removal: {dem / len(sentences)}')

# remove_long_tokenizer(train_data, tokenizer)

# print(train_data['0'].value_counts())

# remove_long_tokenizer(dev_data, tokenizer)

# remove_long_tokenizer(test_data, tokenizer)
from torch.utils.data import Dataset

dir = '../input/pretrainedhoc/bio_roberta-base-ohsumed_v1.0/bio_roberta-base-ohsumed_v1.0'

tokenizer = RobertaTokenizer.from_pretrained(dir, do_lower_case=False)

import random

class TextSequenceDataset(Dataset):

    """ Text Sequence Dataset. """

    def __init__(self, df, tokenizer, max_length=512, is_train=False, transform=None, threshold=0.2):

        if is_train:

            self.df = self.init_df(df, threshold)

        else:

            self.df = df

        self.label_columns = self.df.columns[1:-1]

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.transform = transform

        

    def init_df(self, df, threshold):

        new_df = pd.DataFrame(columns=df.columns)

        for i in range(len(df)):

            check = True

            for j in range(10):

                col = str(j)

                if df.iloc[i][col] != 0:

                    new_df = new_df.append(df.iloc[i])

                    check = False

                    break

            rd = random.uniform(0, 1)

            if rd < threshold:

                new_df = new_df.append(df.iloc[i])

        return new_df

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        text = self.df.iloc[idx]['sentence']

        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_token_type_ids=False)

        labels = np.array(self.df.iloc[idx][self.label_columns], dtype=float)

        sample = {'text':encoding, 'label':labels}

        if self.transform:

            sample = self.transform(sample)

        return sample



class ToTensor(object):

    def __call__(self, sample):

        encoding, label = sample['text'], sample['label']

        if 'input_ids' in encoding.keys():

            encoding['input_ids'] = torch.from_numpy(np.array(encoding['input_ids']))

        if 'attention_mask' in encoding.keys():

            encoding['attention_mask'] = torch.from_numpy(np.array(encoding['attention_mask']))

        if 'token_type_ids' in encoding.keys():

            encoding['token_type_ids'] = torch.from_numpy(np.array(encoding['token_type_ids']))

        return {'text':encoding, 'label': torch.from_numpy(label)}
from torch.utils.data import DataLoader

from transformers import *

import torch.nn as nn
class Multilabel_Cfs(nn.Module):

    def __init__(self, pre_trained='roberta-base', config=None):

        super().__init__()

        if config is not None:

            print('loading')

            self.config = RobertaConfig.from_json_file(dir + '/' + config)

        else:

            self.config = RobertaConfig.from_pretrained(pre_trained)

        self.roberta_model = RobertaModel.from_pretrained(pre_trained)

        self.top_classify = Classify_Layer(self.config, 10)



    def forward(

        self,

        input_ids=None,

        attention_mask=None,

        token_type_ids=None,

        position_ids=None,

        head_mask=None,

        inputs_embeds=None,

        labels=None,

        output_attentions=None,

        output_hidden_states=None,

        return_dict=None,

    ):

        r"""

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):

            Labels for computing the sequence classification/regression loss.

            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.

            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),

            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        outputs = self.roberta_model(

            input_ids,

            attention_mask=attention_mask,

            token_type_ids=token_type_ids,

            position_ids=position_ids,

            head_mask=head_mask,

            inputs_embeds=inputs_embeds,

            output_attentions=output_attentions,

            output_hidden_states=output_hidden_states,

            return_dict=return_dict,

        )

        sequence_output = outputs[0]

        pred = self.top_classify(sequence_output)

        return pred

class Classify_Layer(nn.Module):

    """ Head for multi-label classification. """

    def __init__(self, config, num_label):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, num_label)

        # self.dropout = nn.Dropout(p=0.5)

        # self.out_linear = nn.Linear(config.hidden_size, num_label)

        self.sigmoid = nn.Sigmoid()



    def forward(self, features, **kwargs):

        x = features[:, 0]

        x = self.dense(x)

        # x = self.dropout(x)

        # x = self.out_linear(x)

        x = self.sigmoid(x)

        return x


config_file = 'config.json'

model_cf = Multilabel_Cfs(dir, config=config_file)

model_cf.to(device)
def convert(x):

    """

    Convert x to 0 - 1 value

    """

    for i in range(x.shape[0]):

        for j in range(x.shape[1]):

            if x[i, j] >= 0.5:

                x[i, j] = 1

            else:

                x[i, j] = 0

    return x

tmp = np.random.rand(2,3)

print(tmp)

convert(tmp)
from sklearn.metrics import confusion_matrix

def micro_f1_score_ver_2(output, label, is_print=True, loss=0.0):

    # (0, 0): TN

    # (0, 1): FP

    # (1, 0): FN

    # (1, 1): TP

    precisions, recalls = np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)

    for i in range(10):

        conf = confusion_matrix(label[i], output[i])

        # precision = tp / (tp + fp)

        if conf[1][1] + conf[0][1] != 0:

            precisions[i] = conf[1][1] / (conf[1][1] + conf[0][1])

        else:

            precisions[i] = np.NaN

        if conf[1][1] + conf[1][0] != 0:

            recalls[i] =  conf[1][1] / (conf[1][1] + conf[1][0])

        else:

            recalls[i] = np.NaN



    precision_a, precision_b, recall_a, recall_b = 0, 0, 0, 0

    for i in range(10):

        conf = confusion_matrix(label[i], output[i])

        precision_a += conf[1][1]

        precision_b += conf[1][1] + conf_matrix[i][0][1]

        recall_a += conf[1][1]

        recall_b += conf[1][1] + conf[1][0]

    precision = precision_a/precision_b

    recall = recall_a / recall_b

    f1 = 2 * precision * recall / (recall + precision)

    if is_print:

        print_f1(f1, precision, recall, precisions, recalls, loss)

    return f1, precisions, recalls


from sklearn.metrics import multilabel_confusion_matrix

def micro_f1_score(output, label, is_print=True, loss=0.0):

    # (0, 0): TN

    # (0, 1): FP

    # (1, 0): FN

    # (1, 1): TP

    conf_matrix = multilabel_confusion_matrix(label, output)

    precisions, recalls = np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32)

    for i in range(10):

        conf = conf_matrix[i]

        # precision = tp / (tp + fp)

        if conf[1][1] + conf[0][1] != 0:

            precisions[i] = conf[1][1] / (conf[1][1] + conf[0][1])

        else:

            precisions[i] = np.NaN

        if conf[1][1] + conf[1][0] != 0:

            recalls[i] =  conf[1][1] / (conf[1][1] + conf[1][0])

        else:

            recalls[i] = np.NaN

    precision_a, precision_b, recall_a, recall_b = 0, 0, 0, 0

    for i in range(10):

        precision_a += conf_matrix[i][1][1]

        precision_b += conf_matrix[i][1][1] + conf_matrix[i][0][1]

        recall_a += conf_matrix[i][1][1]

        recall_b += conf_matrix[i][1][1] + conf_matrix[i][1][0]

    precision = precision_a/precision_b

    recall = recall_a / recall_b

    f1 = 2 * precision * recall / (recall + precision)

    if is_print:

        print_f1(f1, precision, recall, precisions, recalls, loss)

    return f1, precisions, recalls

def print_f1(f1, precision, recall, precisions, recalls, loss=0.0):

    if loss != 0.0:

        print(' -- BCE Loss: {:4f}'.format(loss))

    print(' -- Micro_F1_Score: {:.4f} --- {:4f} - {:4f}'.format(f1, precision, recall))

    print(' -- Precision Score: ', end='')

    with np.printoptions(precision=3, suppress=True):

        print(precisions)

    print(' -- Recall Score:    ', end='')

    with np.printoptions(precision=3, suppress=True):

        print(recalls)

# BATCH_SIZE = 32

from barbar import Bar

from tqdm import tqdm

import math

def train(model_cf, dataloader, criterion, optimizer):

    model_cf.train()

    epoch_loss = 0

    y_pred, y_true = np.zeros((0,10)), np.zeros((0,10))



    for i, sample in enumerate(Bar(dataloader)):

        input_ids = sample['text']['input_ids'].to(device)

        attention_mask=sample['text']['attention_mask'].to(device)

        # token_type_ids=sample['text']['token_type_ids'].to(device)

        label = sample['label'].to(device, dtype=torch.float)

        # print(label)

        optimizer.zero_grad()

        outputs = model_cf(input_ids=input_ids, attention_mask=attention_mask)

        # new weighted loss

        loss = criterion(outputs, label)

        loss.backward()

        optimizer.step()



        epoch_loss += loss.item()

        # print(loss.item())

        y_pred = np.concatenate((y_pred, convert(outputs.detach().cpu().numpy())))

        y_true = np.concatenate((y_true, label.detach().cpu().numpy()))



        if i % 30 == 0:

            f1, precisions, recalls = micro_f1_score(y_pred, y_true, is_print=True, loss=(epoch_loss / (i + 1)))

            # print_f1(f1, precisions, recalls)
def evaluate(model_cf, dataloader, criterion):

    model_cf.eval()

    full_loss = 0

    y_pred, y_true = np.zeros((0,10)), np.zeros((0,10))

    for i, sample in enumerate(Bar(dataloader)):

        input_ids = sample['text']['input_ids'].to(device)

        attention_mask=sample['text']['attention_mask'].to(device)

        # token_type_ids=sample['text']['token_type_ids'].to(device)

        label = sample['label'].to(device, dtype=torch.float)        

        with torch.no_grad():

            outputs =  model_cf(input_ids=input_ids, attention_mask=attention_mask)

            full_loss += criterion(outputs, label).item()

            y_pred = np.concatenate((y_pred, convert(outputs.detach().cpu().numpy())))

            y_true = np.concatenate((y_true, label.detach().cpu().numpy()))

        # break

    f1, precisions, recalls = micro_f1_score(y_pred, y_true, is_print=True, loss=(full_loss/len(dataloader)))

    # print_f1(f1, precisions, recalls)
class Weighted_BCELoss(nn.Module):

    def __init__(self, w=(0.01, 0.99)):

        self.w = torch.Tensor().new_tensor([w[0], w[1]], requires_grad=False)

    def forward(self, output, label):

        n_batch, n_label = output.shape[0], output.shape[1]

        n_loss = 0

        for i in range(n_batch):

            for j in range(n_label):

                n_loss += -(self.w[1] * label[i, j] * torch.log(output[i, j]) + self.w[0] * (1 - label[i, j]) * torch.log(1 - output[i ,j]))

        return n_loss / (n_batch * n_label)
N_EPOCHS = 20

from torch.optim import Adam

# preparing data

train_dataset = TextSequenceDataset(train_data, tokenizer, max_length=512, is_train=False, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)



dev_dataset = TextSequenceDataset(dev_data, tokenizer, max_length=512, transform=ToTensor())

dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=0)



def trainIters(model_cf, train_dataloader, dev_dataloader, epochs):

    

    # setting follow roBerta paper

    # init optimizer for warm up

    optimizer = torch.optim.Adam(model_cf.parameters(), lr=5e-6, weight_decay=0.002)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=2)



    criterion = nn.BCELoss(reduction='mean').cuda()

    threshold = epochs * 2 // 10



    for epoch in range(epochs):

        state_lr = optimizer.state_dict()

        lr = state_lr['param_groups'][0]['lr']

        print(f'Epoch: {epoch+1:02} | Learning Rate: {lr}')



        if epoch == threshold:

            optimizer = torch.optim.Adam(model_cf.parameters(), lr=2e-5, weight_decay=0.002)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)



        train(model_cf, train_dataloader, criterion, optimizer)

        evaluate(model_cf, train_dataloader, criterion)

        evaluate(model_cf, dev_dataloader, criterion)

        scheduler.step()
trainIters(model_cf,train_dataloader, dev_dataloader, 20)
# for i, sample in enumerate(Bar(dev_dataloader)):

#     input_ids = sample['text']['input_ids'].to(device)

#     attention_mask=sample['text']['attention_mask'].to(device)

#     label = sample['label'].to(device, dtype=torch.float)  

#     print(input_ids)

#     print(label)      

#     model_cf.eval()

#     with torch.no_grad():

#         # outputs = model_cf.forward(input_ids=input_ids, attention_mask=attention_mask)

#         output2 = model_cf(input_ids=input_ids, attention_mask=attention_mask)

#     break

# print(convert(output2.detach().cpu().numpy()))

# # print(output2)

test_dataset = TextSequenceDataset(test_data, tokenizer, max_length=512, transform=ToTensor())

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
evaluate(model_cf, test_dataloader, nn.BCELoss(reduction='mean').cuda())
torch.save(model_cf.state_dict(), './bio-ohsumed-roberta-hoc.pt')