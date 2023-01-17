!pip install transformers

!pip install keras
import torch 

import torch.nn as nn 

import torch.optim as optim 

import torch.nn.functional as F 

from torch.utils.data import * 

from keras.preprocessing.sequence import pad_sequences 

from transformers import AutoModel,AutoTokenizer

from keras.datasets import imdb 

torch.__version__

import os
MAX_WORDS = 10000  # imdb’s vocab_size 即词汇表大小

MAX_LEN = 512      # max length

BATCH_SIZE = 256

EMB_SIZE = 128   # embedding size

HID_SIZE = 128   # lstm hidden size

DROPOUT = 0.2 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)
def idx2word(idxs):

    word_index = imdb.get_word_index()

    idx_word = dict([(value,key) for (key,value) in word_index.items()])

    sentences = []

    for ids in idxs:

        sentences.append(' '.join([idx_word.get(index-3,'') for index in ids]))

    return sentences
def process(sentences):

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []

    for sentence in sentences:

        input_id = tokenizer.encode(sentence,add_special_tokens=True,max_length=MAX_LEN) #没有padding

        input_ids.append(input_id)

    return input_ids
def padding_mask(input_ids):

    padding_ids = []

    att_masks = []

    for input_id in input_ids:

        padding_ids.append(input_id + (MAX_LEN - len(input_id))*[0])

        att_masks.append([1] * len(input_id) + [0] * (MAX_LEN - len(input_id)))

    return padding_ids,att_masks
(x_train,y_train),(x_val,y_val) = imdb.load_data(num_words=100000)

train_sen = idx2word(x_train)

val_sen = idx2word(x_val)

train_ids_ = process(train_sen)

val_ids_ = process(val_sen)

train_ids,train_masks = padding_mask(train_ids_)

val_ids,val_masks = padding_mask(val_ids_)
# 转化为TensorDataset

train_data = TensorDataset(torch.LongTensor(train_ids),torch.LongTensor(train_masks) ,torch.LongTensor(y_train))

test_data = TensorDataset(torch.LongTensor(val_ids),torch.LongTensor(val_masks) ,torch.LongTensor(y_val))
# 转化为 DataLoader

train_sampler = RandomSampler(train_data)

train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)



test_sampler = SequentialSampler(test_data)

test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
# 定义lstm模型用于文本分类

class Model(nn.Module):

    def __init__(self, hid_size, dropout):

        super(Model, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        for p in self.bert.parameters():

            p.requires_grad = False

        self.hid_size = hid_size

        self.dropout = dropout

        self.emb_size = 768

#         self.Embedding = nn.Embedding(self.max_words, self.emb_size)

        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,

                            batch_first=True, bidirectional=True)   # 2层双向LSTM

        self.dp = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)

        self.fc2 = nn.Linear(self.hid_size, 2)

    

    def forward(self,ids,masks):

        """

        input : [bs, maxlen]

        output: [bs, 2] 

        """

#         x = self.Embedding(x)  # [bs, ml, emb_size]

        with torch.no_grad():

            x,_ = self.bert(ids,masks)

#         print(len(x))

#         x = self.dp(x)

        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]

        x = self.dp(x)

        x = F.relu(self.fc1(x))   # [bs, ml, hid_size]

        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]

        out = self.fc2(x)    # [bs, 2]

        return out  # [bs, 2]
def train(model, device, train_loader, optimizer, epoch):   # 训练模型

    model.train()

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x, y,z) in enumerate(train_loader):

        x, y ,z = x.to(DEVICE), y.to(DEVICE) , z.to(DEVICE)

        optimizer.zero_grad()

        y_ = model(x,y)

        loss = criterion(y_, z)  # 得到loss

        loss.backward()

        optimizer.step()

        if(batch_idx + 1) % 10 == 0:    # 打印loss

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(x), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))
def test(model, device, test_loader):    # 测试模型

    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss

    test_loss = 0.0 

    acc = 0 

    for batch_idx, (x, y,z) in enumerate(test_loader):

        x, y , z = x.to(DEVICE), y.to(DEVICE),z.to(DEVICE)

        with torch.no_grad():

            y_ = model(x,y)

        test_loss += criterion(y_, z)

        pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index

        acc += pred.eq(z.view_as(pred)).sum().item()    # 记得加item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(

        test_loss, acc, len(test_loader.dataset),

        100. * acc / len(test_loader.dataset)))

    return acc / len(test_loader.dataset) 
model = Model( HID_SIZE, DROPOUT).to(DEVICE)

# print(model)

optimizer = optim.Adam(model.parameters())



best_acc = 0.0 

PATH = '/kaggle/working/params.pkl'  # 定义模型保存路径

if os.path.exists(PATH):

    model.load_state_dict(torch.load(PATH))

for epoch in range(1):  # 10个epoch

    train(model, DEVICE, train_loader, optimizer, epoch)

    acc = test(model, DEVICE, test_loader)

    if best_acc < acc: 

        best_acc = acc 

        torch.save(model.state_dict(), PATH)

    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc)) 
# 检验保存的模型

best_model = Model(HID_SIZE, DROPOUT).to(DEVICE)

best_model.load_state_dict(torch.load(PATH))

test(best_model, DEVICE, test_loader)
# only lstm : the acc is about 87%

# bert lstm : about 90%