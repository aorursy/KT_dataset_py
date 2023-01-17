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
from gensim.models import KeyedVectors

from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split

import string

import re

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data

import numpy as np

import pandas as pd
def remove_URL(text):

    url = re.compile(r'https?://\S+')

    return url.sub(r'', text)





def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'', text)





def remove_emoji(text):

    emoji_pattern = re.compile("["

                               u"\U0001F600-\U0001F64F"  # emoticons

                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                               u"\U0001F680-\U0001F6FF"  # transport & map symbols

                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                               u"\U00002702-\U000027B0"

                               u"\U000024C2-\U0001F251"

                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)





def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)





def clean(x):

    # Spelling checking is omitted because it takes much time

    return remove_punct(remove_emoji(remove_html(remove_URL(x))))
class Word2VecVectorizer:

    def __init__(self, model_dir, emb_dim=300, seq_len=256):

        self.max_len = seq_len

        self.emb_dim = emb_dim

        self.model = KeyedVectors.load_word2vec_format(model_dir, binary=True)



    def vectorize(self, sentence):

        words = clean(sentence).split()

        textvec = np.zeros((self.max_len, self.emb_dim))

        for i, word in enumerate(words[:self.max_len]):

            try:

                textvec[i] = self.model[word.lower()]

            except KeyError:

                textvec[i] = np.zeros(self.emb_dim)

        return torch.tensor(textvec).float()





class BertSequenceVectorizer:

    def __init__(self, seq_len):

        self.model_name = 'bert-base-uncased'

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        self.bert_model = BertModel.from_pretrained(self.model_name)

        self.max_len = seq_len



    def vectorize(self, sentence):

        inp = self.tokenizer.encode(clean(sentence))

        len_inp = len(inp)

        if len_inp >= self.max_len:

            inputs = inp[:self.max_len]

            masks = [1] * self.max_len

        else:

            inputs = inp + [0] * (self.max_len - len_inp)

            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long)

        masks_tensor = torch.tensor([masks], dtype=torch.long)

        seq_out, _ = self.bert_model(inputs_tensor, masks_tensor)

        return seq_out[0]
class TweetDataset(data.Dataset):



    def __init__(self, text_list, label_list, transform=None, phase='train'):

        self.text_list = text_list

        self.label_list = label_list

        self.transform = transform

        self.phase = phase



    def __len__(self):

        return len(self.text_list)



    def __getitem__(self, index):

        textvec = self.transform(self.text_list[index])

        label = self.label_list[index]

        return textvec, label
class PositionalEncoder(nn.Module):

    def __init__(self, emb_dim=300, seq_len=256):

        super().__init__()

        self.emb_dim = emb_dim

        pe = torch.zeros(seq_len, emb_dim)

        for pos in range(seq_len):

            for i in range(0, emb_dim, 2):

                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_dim)))

                pe[pos, i] = math.cos(

                    pos / (10000 ** ((2 * (i + 1)) / emb_dim)))

        self.pe = pe.unsqueeze(0)

        self.pe.requires_grad = False



    def forward(self, x):

        ret = math.sqrt(self.emb_dim) * x + self.pe

        return ret
class Attention(nn.Module):

    def __init__(self, emb_dim=300):

        super().__init__()



        self.q_linear = nn.Linear(emb_dim, emb_dim)

        self.v_linear = nn.Linear(emb_dim, emb_dim)

        self.k_linear = nn.Linear(emb_dim, emb_dim)



        self.out = nn.Linear(emb_dim, emb_dim)

        self.emb_dim = emb_dim



    def forward(self, q, k, v):

        k = self.k_linear(k)

        q = self.q_linear(q)

        v = self.v_linear(v)



        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.emb_dim)

        normlized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normlized_weights, v)

        output = self.out(output)



        return output, normlized_weights





class FeedForward(nn.Module):

    def __init__(self, emb_dim, ff_dim=1024, dropout=0.1):

        super().__init__()



        self.linear_1 = nn.Linear(emb_dim, ff_dim)

        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(ff_dim, emb_dim)



    def forward(self, x):

        x = self.linear_1(x)

        x = self.dropout(F.relu(x))

        x = self.linear_2(x)

        return x





class TransformerBlock(nn.Module):

    def __init__(self, emb_dim, dropout=0.1):

        super().__init__()



        self.norm_1 = nn.LayerNorm(emb_dim)

        self.norm_2 = nn.LayerNorm(emb_dim)



        self.attn = Attention(emb_dim)



        self.ff = FeedForward(emb_dim)



        self.dropout_1 = nn.Dropout(dropout)

        self.dropout_2 = nn.Dropout(dropout)



    def forward(self, x):

        x_normlized = self.norm_1(x)

        output, normlized_weights = self.attn(

            x_normlized, x_normlized, x_normlized)



        x2 = x + self.dropout_1(output)



        x_normlized2 = self.norm_2(x2)

        output = x2 + self.dropout_2(self.ff(x_normlized2))



        return output, normlized_weights
class ClassificationHead(nn.Module):

    def __init__(self, emb_dim=300, output_dim=2):

        super().__init__()



        self.linear = nn.Linear(emb_dim, output_dim)



        nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, x):

        x0 = x[:, 0, :]

        out = self.linear(x0)

        return out
class TransformerClassification(nn.Module):

    def __init__(self, emb_dim=300, seq_len=256, output_dim=2):

        super().__init__()



        self.net1 = PositionalEncoder(emb_dim=emb_dim, seq_len=seq_len)

        self.net2_1 = TransformerBlock(emb_dim=emb_dim)

        self.net2_2 = TransformerBlock(emb_dim=emb_dim)

        self.net3 = ClassificationHead(emb_dim=emb_dim, output_dim=output_dim)



    def forward(self, x):

        x1 = self.net1(x)

        x2_1, normlized_weights_1 = self.net2_1(x1)

        x2_2, normlized_weights_2 = self.net2_1(x2_1)

        x3 = self.net3(x2_2)

        return x3, normlized_weights_1, normlized_weights_2
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):



    for epoch in range(num_epochs):



        for phase in ['train', 'val']:

            if phase == 'train':

                net.train()

            else:

                net.eval()



            epoch_loss = 0.0

            epoch_corrects = 0



            for inputs, labels in dataloaders_dict[phase]:

                optimizer.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):

                    outputs, _, _ = net(inputs)



                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)



                    if phase == 'train':

                        loss.backward(retain_graph=True)

                        optimizer.step()



                    epoch_loss += loss.item() * inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels)



            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)



            print("Epoch {}/{} | {:^5} | loss:{:.4f} Acc:{:.4f}".format(

                epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

    return net
MODEL_DIR = "../input/word2vec-google/GoogleNews-vectors-negative300.bin"

TEST_SIZE = 0.1

LEARNING_RATE = 0.001

NUM_EPOCHS = 10

EMB_DIM = 300  # 768

SEQ_LEN = 64  # 4

BATCH_SIZE = 100

TAG_SIZE = 2



df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')



X_train, X_val, y_train, y_val = train_test_split(

    df_train['text'].values.tolist(), df_train['target'].values.tolist(), test_size=TEST_SIZE, random_state=1)



Id_test, X_test, y_test = df_test['id'].values.tolist(), df_test['text'].values.tolist(), [0] * len(df_test)



emb = Word2VecVectorizer(MODEL_DIR, EMB_DIM, SEQ_LEN)

# emb = BertSequenceVectorizer(SEQ_LEN)



train_dataset = TweetDataset(

    text_list=X_train, label_list=y_train, transform=emb.vectorize, phase='train')

val_dataset = TweetDataset(

    text_list=X_val, label_list=y_val, transform=emb.vectorize, phase='val')

test_dataset = TweetDataset(

    text_list=X_test, label_list=y_test, transform=emb.vectorize, phase='test')



train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=BATCH_SIZE, shuffle=False)



dataloaders_dict = {'train': train_dataloader,

                    'val': val_dataloader, 'test': test_dataloader}



net = TransformerClassification(EMB_DIM, SEQ_LEN, TAG_SIZE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)



net = train_model(net, dataloaders_dict, criterion, optimizer, NUM_EPOCHS)
preds_list = []

for inputs, _ in dataloaders_dict['test']:

    outputs, _, _ = net(inputs)

    _, preds = torch.max(outputs, 1)

    preds_list.extend(preds.tolist())
df_submit = pd.DataFrame(np.array([Id_test, preds_list]).T,columns=['id', 'target'])

print(df_submit)
df_submit.to_csv('/kaggle/working/submission.csv', index=False)