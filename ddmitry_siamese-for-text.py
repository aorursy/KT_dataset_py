import os

import numpy as np

import pandas as ps

from pathlib import Path

from collections import OrderedDict



import matplotlib.pyplot as plt

from fastprogress import master_bar, progress_bar



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'[!] Using {DEVICE}')



NUM_WORKERS = os.cpu_count() - 1

print(f'[!] Using cpus - {NUM_WORKERS} for data loaders')



DATA_FOLDER = Path('..') / 'input'

train_df = ps.read_csv(DATA_FOLDER / 'SPAM text message 20170820 - Data.csv')

print(train_df.shape)

train_df.head()
tokenizer = Tokenizer(num_words=10_000)

tokenizer.fit_on_texts(train_df['Message'].values)

word_indices = tokenizer.texts_to_sequences(train_df['Message'].values)
train_df['target'] = train_df['Category'].map({'ham': 1, 'spam': 0})

train_df['tokenized message length'] = [len(item) for item in word_indices]



word_indices = pad_sequences(word_indices, maxlen=20)

print(word_indices.shape)

train_df.head()
print('Mean sentence length is', int(train_df['tokenized message length'].mean()))
class TextDataset(Dataset):

    def __init__(self, word_indices, labels):

        self.word_indices = word_indices

        self.labels = labels

        self.__unique_labels = set(labels)

        self.label2index = {lbl: np.where(labels == lbl)[0] for lbl in self.__unique_labels}

        

    def __len__(self):

        return len(self.word_indices)

    

    def __getitem__(self, idx):

        seq1, label = self.word_indices[idx], self.labels[idx]

        target = np.random.randint(0, 2)  # 0 or 1

        if target == 1:

            pair_idx = np.random.choice(self.label2index[label])

        else:

            other_class = np.random.choice(list(self.__unique_labels - {label}))

            pair_idx = np.random.choice(self.label2index[other_class])

        seq2 = self.word_indices[pair_idx]

        return torch.from_numpy(seq1).long(), torch.from_numpy(seq2).long(), torch.LongTensor([target])

    



class DummyDataset(Dataset):

    def __init__(self, word_indices, labels):

        self.data, self.labels = word_indices, labels

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        return torch.from_numpy(self.data[idx]).long(), torch.LongTensor([self.labels[idx]])
class EvaEmb(nn.Module):

    def __init__(self, 

                 embedding_size: int, 

                 embedding_dim: int, 

                 num_classes: int, 

                 hidden_rnn_size: int = 60, 

                 dropout_rate: float = 0.3, 

                 lstm_layers: int = 1,

                 gru_layers: int = 1):

        super(EvaEmb, self).__init__()

        self.embedding = nn.Embedding(embedding_size, embedding_dim)

        self.lstm = nn.LSTM(

            input_size=embedding_dim,

            hidden_size=hidden_rnn_size,

            bias=True,

            num_layers=lstm_layers,

            batch_first=True,

            dropout=dropout_rate,

            bidirectional=True

        )

        self.gru = nn.GRU(

            input_size=hidden_rnn_size * 2,

            hidden_size=hidden_rnn_size,

            bias=True,

            num_layers=gru_layers,

            batch_first=True,

            dropout=dropout_rate,

            bidirectional=True

        )

        self.max_pool = nn.MaxPool1d(3, stride=2)

        self.avg_pool = nn.AvgPool1d(3, stride=2)

        input_size = (hidden_rnn_size * 2 - 3) // 2 + 1

        input_size *= 2  # 2 pool layers

        self.liear_part = nn.Sequential(

            OrderedDict([

                ('block1', nn.Sequential(

                    nn.Linear(input_size, int(input_size * 0.75)), 

                    nn.ReLU(True),

                    nn.BatchNorm1d(int(input_size * 0.75)),

                    nn.Dropout(dropout_rate)

                )),

                ('head', nn.Linear(int(input_size * 0.75), num_classes))

            ])

        )

        

    def forward(self, input_sequence):

        x = self.embedding(input_sequence)  # batch * seq -> batch * seq * emb_size

        

        x_lstm, _hidden = self.lstm(x)

        x_gru, _hidden = self.gru(x_lstm)

        

        x_max, x_avg = self.max_pool(x_gru)[:, -1], self.avg_pool(x_gru)[:, -1]

        

        classes = self.liear_part(torch.cat((x_max, x_avg), 1))

        return classes

class SiameseNet(nn.Module):

    def __init__(self, embedding_net):

        super(SiameseNet, self).__init__()

        self.embedding_net = embedding_net



    def forward(self, x1, x2):

        output1 = self.embedding_net(x1)

        output2 = self.embedding_net(x2)

        return output1, output2



    def get_embedding(self, x):

        return self.embedding_net(x)
class ContrastiveLoss(nn.Module):

    """

    Contrastive loss

    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

    """



    def __init__(self, margin):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.eps = 1e-9



    def forward(self, output1, output2, target, size_average=True):

        distances = (output2 - output1).pow(2).sum(1)  # squared distances

        losses = 0.5 * (target.float() * distances +

                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        return losses.mean() if size_average else losses.sum()
dataset = TextDataset(word_indices, train_df['target'].values)

data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
emb_model = EvaEmb(10_000, 64, 2, hidden_rnn_size=32, lstm_layers=2, gru_layers=2)

model = SiameseNet(emb_model).to(DEVICE)
loss = ContrastiveLoss(margin=1)

optimizer = optim.Adam(model.parameters(), lr=1e-3 / 2)
def train(model, loss_function, optimizer, num_epochs, train_loader):

    train_losses = []

    epochs = [i for i in range(1, num_epochs + 1)]

    epochs_iterator = master_bar(epochs)

    for epoch_number in epochs_iterator:

        epoch_losses = []

        model.train()

        for data1, data2, lbl in progress_bar(train_loader, parent=epochs_iterator):

            data1, data2, lbl = data1.to(DEVICE), data2.to(DEVICE), lbl.to(DEVICE)

            out1, out2 = model(data1, data2)

            loss = loss_function(out1, out2, lbl)

            epoch_losses.append(loss.item())

            # optimisation

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epochs_iterator.child.comment = f'loss: {loss.item():.10f}'

        epoch_loss = np.mean(epoch_losses)

        train_losses.append(epoch_loss)

        epochs_iterator.first_bar.comment = f'train loss - {epoch_loss}'

    plt.figure(figsize=(12, 15))

    plt.plot(epochs, train_losses)

    plt.title('Train losses')

    return model
model = train(model, loss, optimizer, 300, data_loader)
def get_embs(model, data_loader):

    model.eval()

    embs, lbls = [], []

    with torch.no_grad():

        for data, lbl in progress_bar(data_loader):

            data = data.to(DEVICE)

            out = model.get_embedding(data).cpu().detach().numpy()

            embs.append(out)

            lbls.append(lbl.numpy())

    return np.concatenate(embs), np.concatenate(lbls)
data_for_plot = DummyDataset(word_indices, train_df['target'].values)

data_for_plot_loader = DataLoader(data_for_plot, batch_size=256, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
embs, lbls = get_embs(model, data_for_plot_loader)
labels = train_df['target'].values

plt.figure(figsize=(15, 15))

for _cls, _color in zip([0, 1], ['b', 'r']):

    indices = np.where(labels == _cls)

    tmp = embs[indices]

    plt.plot(tmp[:, 0], tmp[:, 1], '.', color=_color)

plt.legend(['spam', 'ham'])