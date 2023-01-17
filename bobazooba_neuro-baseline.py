# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/dmia-dl-nlp-2019/train.csv')

test = pd.read_csv('/kaggle/input/dmia-dl-nlp-2019/test.csv')
import torch

from torch.utils.data import Dataset, DataLoader



from nltk.tokenize import word_tokenize, wordpunct_tokenize

from tqdm import tqdm



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
def process_text(text):

    

    # просто токенизация текста, то есть разбиение предложения на токены (слова)

    words = wordpunct_tokenize(text.lower())

    

    return words
process_text('красивая мама мыла красивую раму')
# все наши тексты

texts = list(train.question.map(process_text)) + list(test.question.map(process_text))
# соберем словарь встречаемости слов

# просто подсчет сколько раз то или иное слово встретилось в наших текстах



word2freq = {}



for text in texts:

    

    for word in text:

        

        word2freq[word] = word2freq.get(word, 0) + 1
word2index = {'PAD': 0}

vectors = []

    

word2vec_file = open('/kaggle/input/fasttest-common-crawl-russian/cc.ru.300.vec')

    

n_words, embedding_dim = word2vec_file.readline().split()

n_words, embedding_dim = int(n_words), int(embedding_dim)



# Zero vector for PAD

vectors.append(np.zeros((1, embedding_dim)))



progress_bar = tqdm(desc='Read word2vec', total=n_words)



while True:



    line = word2vec_file.readline().strip()



    if not line:

        break

        

    current_parts = line.split()



    current_word = ' '.join(current_parts[:-embedding_dim])



    if current_word in word2freq:



        word2index[current_word] = len(word2index)



        current_vectors = current_parts[-embedding_dim:]

        current_vectors = np.array(list(map(float, current_vectors)))

        current_vectors = np.expand_dims(current_vectors, 0)



        vectors.append(current_vectors)



    progress_bar.update()



progress_bar.close()



word2vec_file.close()



vectors = np.concatenate(vectors)
unk_words = [word for word in word2freq if word not in word2index]

unk_counts = [word2freq[word] for word in unk_words]

n_unk = sum(unk_counts) * 100 / sum(list(word2freq.values()))



sub_sample_unk_words = {word: word2freq[word] for word in unk_words}

sorted_unk_words = list(sorted(sub_sample_unk_words, key=lambda x: sub_sample_unk_words[x], reverse=True))



print('Мы не знаем {:.2f} % слов в датасете'.format(n_unk))

print('Количество неизвестных слов {} из {}, то есть {:.2f} % уникальных слов в словаре'.format(

    len(unk_words), len(word2freq), len(unk_words) * 100 / len(word2freq)))

print('В среднем каждое встречается {:.2f} раз'.format(np.mean(unk_counts)))

print()

print('Топ 5 невошедших слов:')



for i in range(5):

    print(sorted_unk_words[i], 'с количеством вхождениий -', word2freq[sorted_unk_words[i]])
class WordData(Dataset):

    

    def __init__(self, x_data, y_data, word2index, sequence_length=32, pad_token='PAD', verbose=True):

        

        super().__init__()

        

        self.x_data = []

        self.y_data = y_data

        

        self.word2index = word2index

        self.sequence_length = sequence_length

        

        self.pad_token = pad_token

        self.pad_index = self.word2index[self.pad_token]

        

        self.load(x_data, verbose=verbose)

        

    @staticmethod

    def process_text(text):

    

        words = wordpunct_tokenize(text.lower())



        return words

        

    def load(self, data, verbose=True):

        

        data_iterator = tqdm(data, desc='Loading data', disable=not verbose)

        

        for text in data_iterator:

            words = self.process_text(text)

            indexed_words = self.indexing(words)

            self.x_data.append(indexed_words)

    

    def indexing(self, tokenized_text):



        # выбрасываем неизвестные слова и переводим слова в индекс позиций в матрице эмбеддингов



        return [self.word2index[token] for token in tokenized_text if token in self.word2index]

    

    def padding(self, sequence):

        

        # Ограничить длину self.sequence_length

        # если длина меньше максимально - западить



        return sequence[:self.sequence_length] + [self.pad_index] * (self.sequence_length - len(sequence))

    

    def __len__(self):

        

        return len(self.x_data)

    

    def __getitem__(self, idx):

        

        x = self.x_data[idx]

        x = self.padding(x)

        x = torch.Tensor(x).long()

        

        y = self.y_data[idx]

        

        return x, y
x_train, x_validation, y_train, y_validation = train_test_split(train.question, train.main_category, test_size=0.15)



train_dataset = WordData(list(x_train), list(y_train), word2index)

train_loader = DataLoader(train_dataset, batch_size=64)



validation_dataset = WordData(list(x_validation), list(y_validation), word2index)

validation_loader = DataLoader(validation_dataset, batch_size=64)



test_dataset = WordData(list(test.question), np.zeros((test.shape[0])), word2index)

test_loader = DataLoader(test_dataset, batch_size=64)
for x, y in train_loader:

    break
# x - это батч размером 64

x
# чтобы составить матрицу мы отрезали длинные предложения до 32 токенов, а короткие дополнили индексом PAD до нужной длины
x.shape
# наши таргеты

y
n_classes = train.main_category.unique().shape[0]
class DeepAverageNetwork(torch.nn.Module):

    

    def __init__(self, embedding_matrix, n_classes):

        

        super().__init__()

        

        # здесь мы как раз передаем вектора слов в нашу матрицу эмбеддингов

        # по умолчанию метод from_pretrained замораживает эту матрицу

        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix))

        

        self.layers = torch.nn.Sequential(torch.nn.Linear(300, 256),

                                          torch.nn.ReLU(), 

                                          torch.nn.Linear(256, 128),

                                          torch.nn.ReLU(),

                                          torch.nn.Linear(128, n_classes))

    def forward(self, x):

        

        # переводим индексы слов в вектора

        x = self.embedding_layer(x)

        

        # усредняем эмбеддинги слов

        # переходим к одну вектору на предложение

        # обратите внимание, что за счет нулевого токена PAD мы усредняем нечестно, считая, что у всех предложений длина 32 токена

        x = x.mean(dim=-2)

        

        # применяем несколько линейных слоев с релу

        x = self.layers(x)

        

        return x
# инициализируем модель

model = DeepAverageNetwork(embedding_matrix=vectors, n_classes=n_classes)
# смотрим отработает ли наша модель

# нет ли багов

with torch.no_grad():

    pred = model(x)

    

pred.shape
embeddings = model.embedding_layer(x)
# эмбеддинги слов

# 64 - размер батча

# 32 - количество слов в примере

# 300 - размер эмбеддинга на каждое слово

embeddings.shape
# задаем девайс, где будет учиться модель

# если доступна гпу, то зададим гпу

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# напомню, что мы не используем в моделе софтмакс, потому что он уже есть здесь

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters())



model = model.to(device)

criterion = criterion.to(device)
epochs = 10

losses = []

best_test_loss = 10.



test_f1 = []



for n_epoch in range(epochs):

    

    train_losses = []

    test_losses = []

    test_targets = []

    test_pred_class = []

    

    progress_bar = tqdm(total=len(train_loader.dataset), desc='Epoch {}'.format(n_epoch + 1))

    

    model.train()

    

    for x, y in train_loader:



        x = x.to(device)

        y = y.to(device)

        

        optimizer.zero_grad()

        

        pred = model(x)

        loss = criterion(pred, y)

        

        loss.backward()

        

        optimizer.step()

        

        train_losses.append(loss.item())

        losses.append(loss.item())

        

        progress_bar.set_postfix(train_loss = np.mean(losses[-500:]))



        progress_bar.update(x.shape[0])

        

    progress_bar.close()

    

    model.eval()

    

    for x, y in validation_loader:

        

        x = x.to(device)

        y = y.to(device)



        with torch.no_grad():



            pred = model(x)



            pred = pred.cpu()

            y = y.cpu()



            test_targets.append(y.numpy())

            test_pred_class.append(np.argmax(pred, axis=1))



            loss = criterion(pred, y)



            test_losses.append(loss.item())

        

    mean_test_loss = np.mean(test_losses)



    test_targets = np.concatenate(test_targets).squeeze()

    test_pred_class = np.concatenate(test_pred_class).squeeze()



    f1 = f1_score(test_targets, test_pred_class, average='micro')



    test_f1.append(f1)

    

    print()

    print('Losses: train - {:.3f}, test - {:.3f}'.format(np.mean(train_losses), mean_test_loss))



    print('F1 test - {:.3f}'.format(f1))

        

    # наивный early stopping

    if mean_test_loss < best_test_loss:

        best_test_loss = mean_test_loss

    else:

        print('Early stopping')

        break
model.eval()



predictions = []



for x, _ in test_loader:



    x = x.to(device)



    with torch.no_grad():



        pred = model(x)



        pred = pred.cpu()

        

        predictions.append(np.argmax(pred, axis=1))

        

predictions = np.concatenate(predictions).squeeze()
test['main_category'] = predictions
test = test[['index', 'main_category']]
test.head()
test.to_csv('submission.csv', index=False)