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
!unzip /kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip
!unzip /kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip
data = pd.read_csv('train.tsv',sep = '\t')
data.info()
data['Sentiment'].value_counts(normalize = True)
from sklearn.model_selection import train_test_split
train_d, test_d, train_y, test_y = train_test_split(
    data['Phrase'], data['Sentiment'], test_size=0.25, random_state=5)
def get_dummies(labels, size = 5):
    res = []
    for i in labels:
        temp = [0] * size
        temp[i] = 1
        res.append(temp)
    return res

train_labels, test_labels = get_dummies(train_y), get_dummies(test_y)
!pip install pytorch_transformers
from pytorch_transformers import BertTokenizer
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

sample_sentence = train_d[0]

print(sample_sentence)

print(tokenizer.tokenize('[CLS]' + sample_sentence + '[SEP]'))

print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + sample_sentence + '[SEP]')))


# print(tokenizer.encode_plus(
#                         sample_sentence,    # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = 100,           # Pad & truncate all sentences.
#                         pad_to_max_length = True,
#                         return_attention_mask = True,   # Construct attn. masks.
# #                         return_tensors = 'pt',     # Return pytorch tensors.
#                    ))
from pytorch_transformers import BertTokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_text = [tokenizer.tokenize('[CLS]' + i + '[SEP]') for i in train_d]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
len(input_ids)
input_ids[0]
for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) != 128:
        input_ids[j].extend([0] * (128 - len(i))) #extend sentence to 512, but we dont need dat much
from torch.utils.data import DataLoader, TensorDataset
import torch
train_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.FloatTensor(train_labels))
train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle = True)
tokenized_text = [tokenizer.tokenize('[CLS]' + i + '[SEP]') for i in test_d]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) != 128:
        input_ids[j].extend([0] * (128 - len(i))) #extend sentence to 512, but we dont need dat much

test_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.FloatTensor(test_labels))
test_loader = DataLoader(dataset = test_set, batch_size = 32, shuffle = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel

class fn_cls(nn.Module):
    def __init__(self):
        super(fn_cls, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(768, 5) #768 is the bert-base hidden size
    def forward(self, x, attention_mask = None):
        outputs = self.model(x, attention_mask = attention_mask)
        x = outputs[1]
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x        
from pytorch_transformers import BertModel
model = BertModel.from_pretrained(model_name)
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
from torch import optim

cls = fn_cls()
cls.to(device)
lossF = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(cls.parameters(), lr = 1e-5)
def predict(logits):
    res = torch.argmax(logits, 1)
    return res
def train_model(data, target):
    correct = 0
    cls.train()
    data = data.to(device)
    target = target.to(device)
    mask = []
    for sample in data:
        mask.append([1 if i != 0 else 0 for i in sample])
    mask = torch.Tensor(mask).to(device)
    
    output = cls(data, attention_mask = mask)
    loss = lossF(sigmoid(output).view(-1, 5), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    pred = predict(output)
    correct += (pred == predict(target)).sum().item()
    
    return correct,loss
    
def eval_model():
    cls.eval()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)

        output = cls(data, attention_mask=mask)
        pred = predict(output)

        correct += (pred == predict(target)).sum().item()
        total += len(data)
    return correct/total
from torch.autograd import Variable
import time

pre = time.time()
epoch = 3

for i in range(epoch):
    train_corrects = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).view(-1, 5).to(device)
        train_correct,loss = train_model(data, target)
        train_corrects.append(train_correct)
        if batch_idx % 1000 == 0:
            test_correct = eval_model()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss:{:.4f} \tTraining_acc: {:.4f} \tTesting_acc: {:.4f}'.format(
                i+1, batch_idx, len(train_loader), 100. *
                batch_idx/len(train_loader), loss.item(),
                sum(train_corrects)/ ((batch_idx + 1) * 32),
                test_correct))

                   
print('time comsumed: ', time.time() - pre)
test_df = pd.read_csv('test.tsv', sep = '\t')
test_df.info()
len(test_df)
tokenized_text = [tokenizer.tokenize('[CLS]' + i + '[SEP]') for i in test_df['Phrase']]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) != 128:
        input_ids[j].extend([0] * (128 - len(i))) #extend sentence to 512, but we dont need dat much

output_data = TensorDataset(torch.LongTensor(input_ids))
output_loader = DataLoader(dataset = output_data, batch_size = 1, shuffle = False)
from tqdm.notebook import tqdm
cls.eval()

res_pred = []
for batch_idx, (data) in enumerate(tqdm(output_loader)):
    data = data[0].to(device)
    
    mask = []
    for sample in data:
        mask.append([1 if i != 0 else 0 for i in sample])
    mask = torch.Tensor(mask).to(device)
    
    output = cls(data, attention_mask=mask)
    pred = predict(output)
    res_pred.append(pred)
    
res_final = [result.item() for result in res_pred]
test_df['Sentiment'] = res_final
result_df = test_df[['PhraseId','Sentiment']]
input_id
input_ids[0]
result_df.to_csv('Submission.csv',index = False)
