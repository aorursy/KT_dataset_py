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



model_dir = '../input/word2vec-google/GoogleNews-vectors-negative300.bin'

word2vec_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})

print(df_train)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df_train['text'].values.tolist(), df_train['target'].values.tolist(), test_size=0.01, random_state=1)

Id_test = df_test['id'].values.tolist()

X_test = df_test['text'].values.tolist()

y_test = [0]*len(Id_test)
import string

import re



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
import numpy as np

import torch

from torch import from_numpy



VOCAB_SIZE = 15

EMBEDDING_DIM = 300



def text2tensor(text):

    words = clean(text).split()

    textvec = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

    for i, word in enumerate(words[:VOCAB_SIZE]):

        try:

            textvec[i] = word2vec_model[word.lower()]

        except KeyError:

            textvec[i] = np.zeros(EMBEDDING_DIM)

    return torch.tensor(textvec).float()
import torch.utils.data as data



class TweetDataset(data.Dataset):

    

    def __init__(self, text_list, label_list, transform=None, phase='train'):

        self.text_list = text_list

        self.label_list = label_list

        self.transform = transform

        self.phase = phase

    

    def __len__(self):

        return len(self.text_list)

    

    def __getitem__(self, index):

        text = self.text_list[index]

        text_transformed = self.transform(text)

        label = self.label_list[index]

        return text_transformed, label



train_dataset = TweetDataset(text_list=X_train, label_list=y_train, transform=text2tensor, phase='train')

val_dataset = TweetDataset(text_list=X_val, label_list=y_val, transform=text2tensor, phase='val')

test_dataset = TweetDataset(text_list=X_test, label_list=y_test, transform=text2tensor, phase='test')
BATCH_SIZE = 100



train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
import torch.nn as nn



class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, inputs):

        _, lstm_out = self.lstm(inputs)

        tag_space = self.hidden2tag(lstm_out[0])

        tag_scores = self.softmax(tag_space.squeeze())

        return tag_scores
import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm



HIDDEN_DIM = 128

TAG_SIZE = 2  # 0 or 1

model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)



NUM_EPOCHS=7

for epoch in range(NUM_EPOCHS):

    print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))

    for phase in ['train', 'val']:

        if (epoch == 0) and (phase == 'train'):

            continue

        epoch_loss = 0.0

        epoch_corrects = 0

        for inputs, labels in tqdm(dataloaders_dict[phase]):

            model.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            if phase == 'train':

                loss.backward()

                optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

            epoch_corrects += torch.sum(preds == labels)

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print("{} loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

print("done.")
preds_list = []

for inputs, _ in dataloaders_dict['test']:

    outputs = model(inputs)

    _, preds = torch.max(outputs, 1)

    preds_list.extend(preds.tolist())
df_submit = pd.DataFrame(np.array([Id_test, preds_list]).T,columns=['id', 'target'])

print(df_submit)
df_submit.to_csv('/kaggle/working/submission.csv', index=False)