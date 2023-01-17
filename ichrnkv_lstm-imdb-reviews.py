import warnings
warnings.filterwarnings('ignore')

import collections
import numpy as np 
import pandas as pd 
import copy
import re
import traceback

import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# чтение данных

def read_data(path_to_data):
    df = pd.read_csv(path_to_data)
    df['sentiment'] = df['sentiment'].map({'positive':0, 'negative':1})
    return df['review'], df['sentiment']

X, y = read_data('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
# разделение данных на трейн и тест

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.1,
                                                   stratify=y)

print('Number of 1/0 classes elements in train: {}'.format(np.bincount(y_train)))
print('Number of 1/0 classes elements in test: {}'.format(np.bincount(y_test)))
TOKEN_RE = re.compile(r'[\w\d]+')


def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


# токенизируем
train_tokenized = tokenize_corpus(X_train.values)
test_tokenized = tokenize_corpus(X_test.values)
def add_fake_token(word2id, token='<PAD>'):
    word2id_new = {token: i + 1 for token, i in word2id.items()}
    word2id_new[token] = 0
    return word2id_new


def texts_to_token_ids(tokenized_texts, word2id):
    return [[word2id[token] for token in text if token in word2id]
            for text in tokenized_texts]


def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    # убрать слишком редкие и слишком частые слова
    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    # отсортировать слова по убыванию частоты
    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    # добавим несуществующее слово с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # если у нас по прежнему слишком много слов, оставить только max_size самых частотных
    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # нумеруем слова
    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    # нормируем частоты слов
    word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return word2id, word2freq


# строим словарь
vocabulary, word_doc_freq = build_vocabulary(train_tokenized, max_doc_freq=0.9, min_count=5, pad_word='<PAD>')
print("Размер словаря", len(vocabulary))
print(list(vocabulary.items())[:10])
# отображаем в номера токенов
train_token_ids = texts_to_token_ids(train_tokenized, vocabulary)
test_token_ids = texts_to_token_ids(test_tokenized, vocabulary)


print('\n'.join(' '.join(str(t) for t in sent)
                for sent in train_token_ids[:1]))
plt.figure(figsize=(8, 8))

plt.hist([len(s) for s in train_token_ids], bins=100);
plt.xticks(np.arange(0, 6000, 500))
plt.title('Гистограмма длин предложений');
def ensure_length(txt, out_len, pad_value):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return txt


class PaddedSequenceDataset(Dataset):
    def __init__(self, texts, targets, out_len=100, pad_value=0):
        self.texts = texts
        self.targets = targets
        self.out_len = out_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        txt = self.texts[item]

        txt = ensure_length(txt, self.out_len, self.pad_value)
        txt = torch.tensor(txt, dtype=torch.long)

        target = torch.tensor(self.targets[item], dtype=torch.long)

        return txt, target
    
    
MAX_SENTENCE_LEN = 300
train_dataset = PaddedSequenceDataset(train_token_ids,
                                      y_train.values,
                                      out_len=MAX_SENTENCE_LEN)
test_dataset = PaddedSequenceDataset(test_token_ids,
                                     y_test.values,
                                     out_len=MAX_SENTENCE_LEN)
# архитектура нейросети

class Net(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, n_hidden_neurons,
                 n_classes, num_features, num_layers=1):
        super(Net, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_size, n_hidden_neurons, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(n_hidden_neurons*num_features, n_classes)
        
        # self.act = torch.nn.ReLU()
        self.dropout =  torch.nn.Dropout(p=0.3)
        self.sm = torch.nn.Softmax(dim=1)
        
    
    def forward(self, x, hidden=None):
        x = self.emb_layer(x)
        x = self.dropout(x)
        if hidden is not None:
            z, new_hidden = self.lstm(x, n_hidden_neurons)
        else:
            z, new_hidden = self.lstm(x)
        z = torch.flatten(z, start_dim=1)
        next_logits = self.fc(z)
        next_logp = self.sm(next_logits)
        return next_logp, new_hidden
    
    
    def inference(self,x):
        x = self.forward(x)
        return x
model = Net(vocabulary_size=len(vocabulary), embedding_size=500, num_layers=3,
           num_features=MAX_SENTENCE_LEN, n_hidden_neurons=100, n_classes=2)
x_t = None

batch_size = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for x, y in train_dataloader:
    x_t = x
    break
  
"""emb = torch.nn.Embedding(len(vocabulary), 100, padding_idx=0)

x = emb(x_t)

lstm = torch.nn.LSTM(100, 100*4, num_layers=3, batch_first=True)

x, _ = lstm(x)
print(x.shape)

z = torch.flatten(x, start_dim=1)
print(z.shape)"""

z, _ = model(x_t)
batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(vocabulary_size=len(vocabulary), embedding_size=500, num_layers=3,
           num_features=MAX_SENTENCE_LEN, n_hidden_neurons=100, n_classes=2)
model.to(device)

criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train_eval_nn(model,
                  train_dataloader, test_dataloader,
                  loss_function, optimizer,
                  n_epochs, early_stopping_patience,
                  lr=1e-3, l2_reg_alpha=0,
                  scheduler=None, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)
    
    if scheduler:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
        lr_scheduler = scheduler(optimizer)
    else:
        lr_scheduler = None

    
    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(n_epochs):
        try:
            model.train()
            mean_train_loss = 0
            train_batches_n = 0


            for idx, (batch_x, batch_y) in enumerate(train_dataloader):
                if idx > 10000:
                    break
                optimizer.zero_grad()

                x_batch = batch_x.to(device)
                y_batch = batch_y.to(device)
                preds, _ = model(x_batch)
                loss_val = loss_function(preds, y_batch)
                # optimizer.zero_grad()

                loss_val.backward()
                optimizer.step()

                mean_train_loss += float(loss_val)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Среднее значение функции потерь на обучении', mean_train_loss)


            model.eval()               
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for idx, (batch_x, batch_y) in enumerate(test_dataloader):
                    if idx > 1000:
                        break

                    x_batch = batch_x.to(device)
                    y_batch = batch_y.to(device)

                    preds, _ = model.forward(x_batch)
                    loss_val = loss_function(preds, y_batch)

                    mean_val_loss += float(loss_val)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            
            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
                
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                early_stopping_patience))
                break
            
            if lr_scheduler:
                lr_scheduler.step(mean_val_loss)
                
            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
                
        except Exception as ex:
            print('Ошибка при обучении {}\n{}'.format(ex, traceback.format_exc()))
            break
            
    return best_val_loss, best_model



best_val_loss, best_model = train_eval_nn(model=model, device=device,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_function=criterion, optimizer=optimizer,
              n_epochs=100, early_stopping_patience=10,
               lr=1e-3, l2_reg_alpha=0,
              scheduler=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True))
                                          # torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.3))
def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = batch_x.to(device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred, _ = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)
    
    
test_pred = predict_with_model(best_model, test_dataset)
print('Доля верных ответов', accuracy_score(y_test.values, test_pred.argmax(-1)))