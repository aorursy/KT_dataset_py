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
train = pd.read_csv('/kaggle/input/deepnlp-hse-course/train.csv')

test = pd.read_csv('/kaggle/input/deepnlp-hse-course/test.csv')
import torch

from torch.utils.data import Dataset, DataLoader



from nltk.tokenize import word_tokenize, wordpunct_tokenize

from tqdm import tqdm



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
def process_text(text):

    

    words = wordpunct_tokenize(text.lower())

    

    return words
word2freq = {}



for question in tqdm(train.question):

    

    words = process_text(question)

    

    for word in words:

        

        if word in word2freq:

            word2freq[word] += 1

        else:

            word2freq[word] = 1
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



    progress_bar.update(1)



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

        

        # Место для вашей предобработки

    

        words = wordpunct_tokenize(text.lower())



        return words

        

    def load(self, data, verbose=True):

        

        data_iterator = tqdm(data, desc='Loading data', disable=not verbose)

        

        for text in data_iterator:

            words = self.process_text(text)

            indexed_words = self.indexing(words)

            self.x_data.append(indexed_words)

    

    def indexing(self, tokenized_text):



        # здесь мы не используем токен UNK, потому что мы мы его специально не учили

        # становится непонятно какой же эмбеддинг присвоить неизвестному слову,

        # поэтому просто выбрасываем наши неизветсные слова

        

        ### CODE ###



        return [self.word2index[token] for token in tokenized_text if token in self.word2index ]

    

    def padding(self, sequence):

        

        # Ограничить длину self.sequence_length

        # если длина меньше максимально - западить

        

        ### CODE ###



        if len(sequence) > self.sequence_length:

            sequence = sequence[:self.sequence_length]

        elif len(sequence) < self.sequence_length:

            sequence = sequence + [self.pad_index] * (self.sequence_length - len(sequence))



        return sequence

    

    def __len__(self):

        

        return len(self.x_data)

    

    def __getitem__(self, idx):

        

        x = self.x_data[idx]

        x = self.padding(x)

        x = torch.Tensor(x).long()

        

        y = self.y_data[idx]

        

        return x, y
x_train, x_validation, y_train, y_validation = train_test_split(train.question, train.main_category, test_size=0.1)



train_dataset = WordData(list(x_train), list(y_train), word2index)

train_loader = DataLoader(train_dataset, batch_size=64)



validation_dataset = WordData(list(x_validation), list(y_validation), word2index)

validation_loader = DataLoader(validation_dataset, batch_size=64)



test_dataset = WordData(list(test.question), np.zeros((test.shape[0])), word2index)

test_loader = DataLoader(test_dataset, batch_size=64)
for x, y in test_loader:

    break
x.shape
y
n_classes = train.main_category.unique().shape[0]
class DeepAverageNetwork(torch.nn.Module):

    

    def __init__(self, embedding_matrix, n_classes):

        

        super().__init__()

        

        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix))

        

        self.layers = torch.nn.Sequential(torch.nn.Linear(300, 256),

                                          torch.nn.ReLU(), 

                                          torch.nn.Linear(256, 128),

                                          torch.nn.ReLU(),

                                          torch.nn.Linear(128, n_classes))

    def forward(self, x):

        

        sequence_lengths = (x > 0).sum(dim=1)

        sequence_lengths[sequence_lengths == 0.] = 1

        

        x = self.embedding_layer(x)

        

        x = x.mean(dim=-2)

        

        lengths_scaling = sequence_lengths.float() / x.size(1)

        lengths_scaling = lengths_scaling.unsqueeze(1).repeat((1, x.size(-1)))

        x /= lengths_scaling.to(x.device)

        

        x = self.layers(x)

        

        return x
model = DeepAverageNetwork(embedding_matrix=vectors, n_classes=n_classes)
with torch.no_grad():

    pred = model(x)
pred.shape
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

        

    # Early stopping:

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