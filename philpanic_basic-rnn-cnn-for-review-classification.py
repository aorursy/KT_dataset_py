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
#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""

Created on 2020/6/12 15:25

@author: phil

"""



import pandas as pd

import os

import numpy as np

import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader





def prepare_data(dataset_path, sent_col_name, label_col_name):

    """

        读出tsv中的句子和标签

    """

    file_path = os.path.join(dataset_path, "train.tsv.zip")

    data = pd.read_csv(file_path, sep="\t")

    X = data[sent_col_name].values

    y = data[label_col_name].values

    return X, y





class Language:

    """ 根据句子列表建立词典并将单词列表转换为数值型表示 """

    def __init__(self):

        self.word2id = {}

        self.id2word = {}



    def fit(self, sent_list):

        vocab = set()

        for sent in sent_list:

            vocab.update(sent.split(" "))

        word_list = ["<pad>", "<unk>"] + list(vocab)

        self.word2id = {word: i for i, word in enumerate(word_list)}

        self.id2word = {i: word for i, word in enumerate(word_list)}



    def transform(self, sent_list, reverse=False):

        sent_list_id = []

        word_mapper = self.word2id if not reverse else self.id2word

        unk = self.word2id["<unk>"] if not reverse else None

        for sent in sent_list:

            sent_id = list(map(lambda x: word_mapper.get(x, unk), sent.split(" ") if not reverse else sent))

            sent_list_id.append(sent_id)

        return sent_list_id





class ClsDataset(Dataset):

    """ 文本分类数据集 """

    def __init__(self, sents, labels):

        self.sents = sents

        self.labels = labels



    def __getitem__(self, item):

        return self.sents[item], self.labels[item]



    def __len__(self):

        return len(self.sents)





def collate_fn(batch_data):

    """ 自定义一个batch里面的数据的组织方式 """

    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True)



    sents, labels = zip(*batch_data)

    sents_len = [len(sent) for sent in sents]

    sents = [torch.LongTensor(sent) for sent in sents]

    padded_sents = pad_sequence(sents, batch_first=True, padding_value=0)



    return torch.LongTensor(padded_sents), torch.LongTensor(labels),  torch.FloatTensor(sents_len)





def get_wordvec(word2id, vec_file_path, vec_dim=50):

    """ 读出txt文件的预训练词向量 """

    print("开始加载词向量")

    word_vectors = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))

    word_vectors[0, :] = 0  # <pad>

    found = 0

    with open(vec_file_path, "r", encoding="utf-8") as f:

        lines = f.readlines()

        for line in lines:

            splited = line.split(" ")

            if splited[0] in word2id:

                found += 1

                word_vectors[word2id[splited[0]]] = torch.tensor(list(map(lambda x: float(x), splited[1:])))

            if found == len(word2id) - 1:  # 允许<unk>找不到

                break

    print("总共 %d个词，其中%d个找到了对应的词向量" % (len(word2id), found))

    return word_vectors.float()





def make_dataloader(dataset_path="/kaggle/input/sentiment-analysis-on-movie-reviews", sent_col_name="Phrase", label_col_name="Sentiment", batch_size=32, vec_file_path="/kaggle/input/glove6b50dtxt/glove.6B.50d.txt", debug=False):

    # X, y = prepare_datapairs(dataset_path="../dataset/imdb", sent_col_name="review", label_col_name="sentiment")

    X, y = prepare_data(dataset_path=dataset_path, sent_col_name=sent_col_name, label_col_name=label_col_name)



    if debug:

        X, y = X[:100], y[:100]



    X_language = Language()

    X_language.fit(X)

    X = X_language.transform(X)



    word_vectors = get_wordvec(X_language.word2id, vec_file_path=vec_file_path, vec_dim=50)

    # 总共 18229个词，其中12769个找到了对应的词向量 word_vectors = get_wordvec(X_language.word2id,

    # vec_file_path=r"F:\NLP-pretrained-model\glove.twitter.27B\glove.twitter.27B.50d.txt", vec_dim=50)



    # 测试

    # print(X[:2])

    # X_id = X_language.transform(X[:2])

    # print(X_language.transform(X_id, reverse=True))



    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    cls_train_dataset, cls_val_dataset = ClsDataset(X_train, y_train), ClsDataset(X_val, y_val)

    cls_train_dataloader = DataLoader(cls_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    cls_val_dataloader = DataLoader(cls_val_dataset, batch_size=batch_size, collate_fn=collate_fn)



    return cls_train_dataloader, cls_val_dataloader, word_vectors, X_language

#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""

Created on 2020/5/15 22:23

@author: phil

"""

import torch.nn as nn

import torch

import torch.nn.functional as F





class TextRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_of_class, weights=None, rnn_type="RNN", device="cpu"):

        super(TextRNN, self).__init__()



        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.num_of_class = num_of_class

        self.embedding_dim = embedding_dim

        self.rnn_type = rnn_type



        if weights is not None:

            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=weights).to(device)

        else:

            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)



        if rnn_type == "RNN":

            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True).to(device)

            self.hidden2label = nn.Linear(hidden_size, num_of_class).to(device)

        elif rnn_type == "LSTM":

            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True).to(device)

            self.hidden2label = nn.Linear(hidden_size*2, num_of_class).to(device)



    def forward(self, input_sents):

        # input_sents (batch_size, seq_len)

        batch_size, seq_len = input_sents.shape

        # (batch_size, seq_len, embedding_dim)

        embed_out = self.embed(input_sents)



        if self.rnn_type == "RNN":

            h0 = torch.randn(1, batch_size, self.hidden_size).to(device)

            _, hn = self.rnn(embed_out, h0)

        elif self.rnn_type == "LSTM":

            h0, c0 = torch.randn(2, batch_size, self.hidden_size).to(device), torch.randn(2, batch_size, self.hidden_size).to(device)

            output, (hn, _) = self.lstm(embed_out, (h0, c0))

            hn = hn.reshape(batch_size, -1)



        logits = self.hidden2label(hn).squeeze(0)



        return logits





class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_of_class, embedding_vectors=None, kernel_num=100, kerner_size=[3, 4, 5], dropout=0.5, device="cpu"):

        super(TextCNN, self).__init__()

        if embedding_vectors is None:

            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)

        else:

            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=embedding_vectors).to(device)

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embedding_dim)).to(device) for K in kerner_size])

        self.dropout = nn.Dropout(dropout).to(device)

        self.feature2label = nn.Linear(3*kernel_num, num_of_class).to(device)



    def forward(self, x):

        # x shape (batch_size, seq_len)

        embed_out = self.embed(x).unsqueeze(1)

        conv_out = [F.relu(conv(embed_out)).squeeze(3) for conv in self.convs]



        pool_out = [F.max_pool1d(block, block.size(2)).squeeze(2) for block in conv_out]



        pool_out = torch.cat(pool_out, 1)



        logits = self.feature2label(pool_out)



        return logits
#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""

Created on 2020/4/30 8:33

@author: phil

"""

from torch import optim

import torch

#from models import TextRNN, TextCNN

# from dataloader_bytorchtext import dataset2dataloader

#from dataloader_byhand import make_dataloader

import torch.nn.functional as F

import numpy as np



if __name__ == "__main__":

    model_names = ["RNN", "LSTM", "CNN"]

    train_iter, val_iter, word_vectors, X_lang = make_dataloader(batch_size=256)

    # train_iter, val_iter, word_vectors = dataset2dataloader(batch_size=128)



    learning_rate = 0.001

    epoch_num = 50

    num_of_class = 5

    MAX_LENGTH = 40

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    

    for model_name in model_names:

        if model_name == "RNN":

            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=128, num_of_class=num_of_class, weights=word_vectors, device=device)

        elif model_name == "CNN":

            model = TextCNN(vocab_size=len(word_vectors), embedding_dim=50, num_of_class=num_of_class, embedding_vectors=word_vectors, device=device)

        elif model_name == "LSTM":

            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=128, num_of_class=num_of_class, weights=word_vectors, rnn_type="LSTM", device=device)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_fun = torch.nn.CrossEntropyLoss()



        for epoch in range(epoch_num):

            model.train()

            for i, batch in enumerate(train_iter):

                # torchtext

                # x, y = batch.review.t()[:, :MAX_LENGTH], batch.sentiment

                x, y, lens = batch

                x = x.to(device)

                y = y.to(device)

                logits = model(x)

                optimizer.zero_grad()

                loss = loss_fun(logits, y)

                loss.backward()

                optimizer.step()

            

            model.eval()

            train_accs = []

            for i, batch in enumerate(train_iter):

                # x, y = batch.review.t()[:, :MAX_LENGTH], batch.sentiment

                x, y, lens = batch

                x = x.to(device)

                y = y.to(device)

                logits = model(x)

                _, y_pre = torch.max(logits.cpu(), -1)

                acc = torch.mean((torch.tensor(y_pre.cpu() == y.cpu(), dtype=torch.float)))

                train_accs.append(acc.cpu())

            train_acc = np.array(train_accs).mean()



            val_accs = []

            for i, batch in enumerate(val_iter):

                # x, y = batch.review.t()[:, :MAX_LENGTH], batch.sentiment

                x, y, lens = batch

                x = x.to(device)

                y = y.to(device)

                logits = model(x)

                _, y_pre = torch.max(logits, -1)

                acc = torch.mean((torch.tensor(y_pre.cpu() == y.cpu(), dtype=torch.float)))

                val_accs.append(acc.cpu())

            val_acc = np.array(val_accs).mean()

            print("epoch %d train acc:%.2f, val acc:%.2f" % (epoch, train_acc, val_acc))

"""

RNN 

epoch 49 train acc:0.51, val acc:0.50

    

LSTM

epoch 49 train acc:0.52, val acc:0.51



CNN

epoch 8 train acc:0.83, val acc:0.67

"""