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
# импортируем необходимые пакеты

import numpy as np 

import pandas as pd 

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.model_selection import train_test_split

import time



import torch

from torchtext import data

import torch.nn as nn

import matplotlib.pyplot as plt

# загружаем обучающие и тестовые данные

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# Посмотрим на наши данныне

train_df[train_df["target"] == 0]["text"].values[10] # твиты не катастрофы
# с рил катастрофой

train_df[train_df["target"] == 1]["text"].values[1]
count_vectorizer = feature_extraction.text.CountVectorizer()



# получим кол-во слов для первых 5 твитов

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])





test_vectors = count_vectorizer.transform(test_df["text"])
# построим простой классификатор и проверим ее качечество, с помощью перекрестной проверки и метрики f1

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
# размер обучающей выборки

train_df.shape
# взгянем на первые 5 строк

train_df.head()
train_df.head()
# Почистим текст

def normalise_text(text):

    text = text.str.lower() 

    text = text.str.replace(r"\#","") 

    text = text.str.replace(r"http\S+","URL")  

    text = text.str.replace(r"@","")

    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")

    text = text.str.replace("\s{2,}", " ")

    return text
train_df["text"]=normalise_text(train_df["text"])
# посмотрим на очищенные данные

train_df['text'].head()
# разделим наш набор на обучающий и проверочный

train_df, valid_df = train_test_split(train_df)
train_df.head()
valid_df.head()
SEED = 42



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
TEXT = data.Field(tokenize = 'spacy', include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)
class DataFrameDataset(data.Dataset):



    def __init__(self, df, fields, is_test=False, **kwargs):

        examples = []

        for i, row in df.iterrows():

            label = row.target if not is_test else None

            text = row.text

            examples.append(data.Example.fromlist([text, label], fields))



        super().__init__(examples, fields, **kwargs)



    @staticmethod

    def sort_key(ex):

        return len(ex.text)



    @classmethod

    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):

        train_data, val_data, test_data = (None, None, None)

        data_field = fields



        if train_df is not None:

            train_data = cls(train_df.copy(), data_field, **kwargs)

        if val_df is not None:

            val_data = cls(val_df.copy(), data_field, **kwargs)

        if test_df is not None:

            test_data = cls(test_df.copy(), data_field, True, **kwargs)



        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
fields = [('text',TEXT), ('label',LABEL)]



train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)
# посмотрим на случайный пимер

print(vars(train_ds[15]))



# тип

print(type(train_ds[15]))
MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(train_ds, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = 'glove.6B.200d',

                 unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds)
# строим итератор

BATCH_SIZE = 128



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_ds, val_ds), 

    batch_size = BATCH_SIZE,

    sort_within_batch = True,

    device = device)
# гиперпараметры

num_epochs = 25

learning_rate = 0.001



INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 200

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.2

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 
# строим модель

class LSTM_net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.rnn = nn.LSTM(embedding_dim, 

                           hidden_dim, 

                           num_layers=n_layers, 

                           bidirectional=bidirectional, 

                           dropout=dropout)

        

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        

        self.fc2 = nn.Linear(hidden_dim, 1)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text, text_lengths):

        

        # text = [sent len, batch size]

        

        embedded = self.embedding(text)

        

        # embedded = [sent len, batch size, emb dim]

        

        #последовательность токенов

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        

        #последовательность распаковки

        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)



        # output = [sent len, batch size, hid dim * num directions]

        # output over padding tokens are zero tensors

        

        # hidden = [num layers * num directions, batch size, hid dim]

        # cell = [num layers * num directions, batch size, hid dim]

        

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers

        # and apply dropout

        

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        output = self.fc1(hidden)

        output = self.dropout(self.fc2(output))

                

        #hidden = [batch size, hid dim * num directions]

            

        return output
model = LSTM_net(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
model
# загрузка предварительно обученных векторов в матрицу вложения

pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)


model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
model.to(device) #gpu





# функция потерь и оптимизатор

criterion = nn.BCEWithLogitsLoss()



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def binary_accuracy(preds, y):

    



    # округляем до целого числа

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float()  

    acc = correct.sum() / len(correct)

    return acc
# создадим функцию для обучения 

def train(model, iterator):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        text, text_lengths = batch.text

        

        optimizer.zero_grad()

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)



        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        



    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator):

    

    epoch_acc = 0

    model.eval()

    

    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            acc = binary_accuracy(predictions, batch.label)

            

            epoch_acc += acc.item()

        

    return epoch_acc / len(iterator)
t = time.time()

loss=[]

acc=[]

val_acc=[]



for epoch in range(num_epochs):

    

    train_loss, train_acc = train(model, train_iterator)

    valid_acc = evaluate(model, valid_iterator)

    

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Acc: {valid_acc*100:.2f}%')

    

    loss.append(train_loss)

    acc.append(train_acc)

    val_acc.append(valid_acc)

    

print(f'time:{time.time()-t:.3f}')
# посмотрим на графике сходимость

plt.xlabel("runs")

plt.ylabel("normalised measure of loss/accuracy")

x_len=list(range(len(acc)))

plt.axis([0, max(x_len), 0, 1])

plt.title('result of LSTM')

loss=np.asarray(loss)/max(loss)

plt.plot(x_len, loss, 'r.',label="loss")

plt.plot(x_len, acc, 'b.', label="accuracy")

plt.plot(x_len, val_acc, 'g.', label="val_accuracy")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)

plt.show