import gensim

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from collections import Counter

from torch.utils.data import TensorDataset,DataLoader

from tqdm import tqdm

from sklearn import metrics
import random

import os

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(323)
class Config:

    TRAIN_PATH = '../input/movie-comment/train.txt'

    VALID_PATH = '../input/movie-comment/validation.txt'

    SAVE_FILE = 'word2id.txt'

    TEST_PATH = '../input/movie-comment/test.txt'

    word2vec_path = '../input/movie-comment/wiki_word2vec_50.bin'

    MAX_LEN = 50

    batch_size = 32

    epochs = 4

    MODEL_PATH = 'model.pth'
def build_word2id(file):

    """

    :param file: word2id保存地址

    :return: None

    """

    word2id = {'_PAD_': 0}

    path = [Config.TRAIN_PATH, Config.VALID_PATH]

    

    for _path in path:

        with open(_path, encoding='utf-8') as f:

            for line in f.readlines():

                sp = line.strip().split()

                for word in sp[1:]:

                    if word not in word2id.keys():

                        word2id[word] = len(word2id)

                      

    with open(file, 'w', encoding='utf-8') as f:

        for w in word2id:

            f.write(w+'\t')

            f.write(str(word2id[w]))

            f.write('\n')

    

    return word2id
def build_word2vec(fname, word2id, save_to_path=None):

    """

    :param fname: 预训练的word2vec.

    :param word2id: 语料文本中包含的词汇集.

    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地

    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.

    """

    n_words = max(word2id.values()) + 1

    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)

    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))

    for word in word2id.keys():

        try:

            word_vecs[word2id[word]] = model[word]

        except KeyError:

            pass

    if save_to_path:

        with open(save_to_path, 'w', encoding='utf-8') as f:

            for vec in word_vecs:

                vec = [str(w) for w in vec]

                f.write(' '.join(vec))

                f.write('\n')

    return word_vecs
def load_corpus(path, word2id, max_len=50, classes=['0', '1']):

    """

    :param path: 样本语料库的文件

    :return: 文本内容contents，以及分类标签labels(onehot形式)

    """

    contents, labels = [], []

    with open(path, encoding='utf-8') as f:

        for line in f.readlines():

            if line.isspace():

                continue

            sp = line.strip().split()

            label = sp[0]

            content = [word2id.get(w, 0) for w in sp[1:]]

            content = content[:max_len]

            if len(content) < max_len:

                content += [word2id['_PAD_']] * (max_len - len(content))

            labels.append(label)

            contents.append(content)



    contents = np.asarray(contents)

    

    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}

    labels = np.array([cat2id[l] for l in labels])



    return contents, labels
class TextCNN(nn.Module):

    def __init__(self,word2vec):

        super(TextCNN, self).__init__()

        

        # 使用预训练的词向量

        self.embedding = nn.Embedding(word2vec.shape[0], word2vec.shape[1])

        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))

        self.embedding.weight.requires_grad = True

        # 卷积层

        self.conv = nn.Conv2d(1,256,(3,50))

        # Dropout

        self.dropout = nn.Dropout(0.2)

        # 全连接层

        self.fc = nn.Linear(256, 2)



    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = F.relu(self.conv(x)).squeeze(3)

        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        x = self.dropout(x)

        x = self.fc(x)

        return x
def train_epoch(data_loader, model, optimizer, device):

    model.train()

    criterion = nn.CrossEntropyLoss()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, (batch_x, batch_y) in enumerate(tk0):

        

        batch_x = batch_x.to(device, dtype=torch.int64)

        batch_y = batch_y.to(device, dtype=torch.int64)

        

        model.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        

        loss.backward()

        optimizer.step()
def eval_epoch(data_loader, model, device):

    model.eval()

    fin_targets = []

    fin_outputs = []

    with torch.no_grad():

        for bi, (batch_x, batch_y) in tqdm(enumerate(data_loader), total=len(data_loader)):

        

            batch_x = batch_x.to(device, dtype=torch.int64)

            batch_y = batch_y.to(device, dtype=torch.int64)



            outputs = model(batch_x)

            

            fin_targets.extend(batch_y.cpu().detach().numpy().tolist())

            fin_outputs.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist())

            

    return fin_outputs, fin_targets

word2id = build_word2id(Config.SAVE_FILE)

word2vec = build_word2vec(Config.word2vec_path, word2id)



train_contents, train_labels = load_corpus(Config.TRAIN_PATH, word2id, Config.MAX_LEN)

val_contents, val_labels = load_corpus(Config.VALID_PATH, word2id, Config.MAX_LEN)

test_contents, test_labels = load_corpus(Config.TEST_PATH, word2id, Config.MAX_LEN)



train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float), 

                              torch.from_numpy(train_labels).type(torch.long))

train_dataloader = DataLoader(dataset = train_dataset, batch_size = Config.batch_size, 

                              shuffle = True, num_workers = 2)



valid_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float), 

                              torch.from_numpy(val_labels).type(torch.long))

valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = Config.batch_size, 

                              shuffle = True, num_workers = 2)



test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float), 

                              torch.from_numpy(test_labels).type(torch.long))

test_dataloader = DataLoader(dataset = test_dataset, batch_size = Config.batch_size, 

                              shuffle = True, num_workers = 2)
print(len(train_dataset))

print(len(valid_dataset))

print(len(test_dataset))
print(Counter(train_labels))

print(Counter(val_labels))

print(Counter(test_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextCNN(word2vec)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
best_accuracy = 0

for epoch in range(Config.epochs):

    train_epoch(train_dataloader, model, optimizer, device)

    outputs, targets = eval_epoch(valid_dataloader, model, device)

    accuracy = metrics.accuracy_score(targets, outputs)

    print(f"Accuracy Score = {accuracy}")

    if accuracy > best_accuracy:

        torch.save(model.state_dict(), Config.MODEL_PATH)

        best_accuracy = accuracy
#测试

outputs, targets = eval_epoch(test_dataloader, model, device)

accuracy = metrics.accuracy_score(targets, outputs)

print(f"Accuracy Score = {accuracy}")