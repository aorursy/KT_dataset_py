import numpy as np

import pandas as pd

import os

import random



from scipy.sparse import coo_matrix, vstack

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data



import nltk

from collections import Counter

import itertools



## --------------------------------------



class Vocab:

    def __init__(self, itos, unk_index):

        self._itos = itos

        self._stoi = {word:i for i, word in enumerate(itos)}

        self._unk_index = unk_index

        

    def __len__(self):

        return len(self._itos)

    

    def word2id(self, word):

        idx = self._stoi.get(word)

        if idx is not None:

            return idx

        return self._unk_index

    

    def id2word(self, idx):

        return self._itos[idx]



class TextToIdsTransformer:

    def transform():

        raise NotImplementedError()

        

    def fit_transform():

        raise NotImplementedError()



class SimpleTextTransformer(TextToIdsTransformer):

    def __init__(self, max_vocab_size):

        self.special_words = ['<PAD>', '</UNK>', '<S>', '</S>']

        self.unk_index = 1

        self.pad_index = 0

        self.vocab = None

        self.max_vocab_size = max_vocab_size

        

    def tokenize(self, text):

        return nltk.tokenize.word_tokenize(text.lower())

        

    def build_vocab(self, tokens):

        itos = []

        itos.extend(self.special_words)

        

        token_counts = Counter(tokens)

        for word, _ in token_counts.most_common(self.max_vocab_size - len(self.special_words)):

            itos.append(word)

            

        self.vocab = Vocab(itos, self.unk_index)

    

    def transform(self, texts):

        result = []

        for text in texts:

            tokens = ['<S>'] + self.tokenize(text) + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

    

    def fit_transform(self, texts):

        result = []

        tokenized_texts = [self.tokenize(text) for text in texts]

        self.build_vocab(itertools.chain(*tokenized_texts))

        for tokens in tokenized_texts:

            tokens = ['<S>'] + tokens + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result



def features_to_tensor(list_of_features):

    text_tensor = torch.tensor([[example.input_ids for example in list_of_features]], dtype=torch.long)

    labels_tensor = torch.tensor([[example.label_id for example in list_of_features]], dtype=torch.long)

    return text_tensor, labels_tensor



## --------------------------------------



def pad(token_ids, max_seq_len, pad_index):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [pad_index for _ in range(max_seq_len - len(token_ids))]

    

    return ids



def get_data():

    dataframe = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv', encoding = 'ISO-8859-1')

    data = dataframe.values



    return data



def remove_unlabeled(x):

    indices = [i for i in range(x.shape[0]) if x[i, 3] != 'pos' and x[i, 3] != 'neg']

    return np.delete(x, indices, 0)



def unrandomize():

    seed = 1448

    

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)



class TextClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, num_filters, num_classes, window_sizes = (3, 4, 5)):

        super(TextClassifier, self).__init__()



        self.embedding = nn.Embedding(vocab_size, emb_size)

        

        self.dropout_big = torch.nn.Dropout(p = 0.4)

        self.dropout_small = torch.nn.Dropout(p = 0.05)



        self.convs = nn.ModuleList([

            nn.Conv1d(1, num_filters, [window_size, emb_size], padding = (window_size - 1, 0)) for window_size in window_sizes

        ])



        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)



    def forward(self, x):

        x = self.embedding(x)

        x = self.dropout_big(x)



        x = torch.unsqueeze(x, 1)

        xs = []

        for i in range(len(self.convs)):

            conv = self.convs[i]

            

            x2 = F.relu(conv(x))

            x2 = torch.squeeze(x2, -1)

            x2 = F.max_pool1d(x2, x2.size(2))

            

            if i == 1:

                x2 = self.dropout_small(x2)

            

            xs.append(x2)

        x = torch.cat(xs, 2)



        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        

        return logits



# class TextClassifier(nn.Module):

#     def __init__(self, vocab_size, emb_size, num_filters, num_classes, window_sizes=(3, 4, 5)):

#         super(TextClassifier, self).__init__()



#         self.embedding = nn.Embedding(vocab_size, emb_size)

        

#         self.dropout = torch.nn.Dropout(p = 0.5)



#         self.convs = nn.ModuleList([

#             nn.Conv2d(1, num_filters, [window_size, emb_size], padding = (window_size - 1, 0)) for window_size in window_sizes

#         ])



#         self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)



#     def forward(self, x):

#         x = self.embedding(x)



#         x = torch.unsqueeze(x, 1)

#         xs = []

#         for conv in self.convs:

#             x2 = F.relu(conv(x))

#             x2 = torch.squeeze(x2, -1)

#             x2 = F.max_pool1d(x2, x2.size(2))

#             xs.append(x2)

#         x = torch.cat(xs, 2)



#         x = x.view(x.size(0), -1)

#         logits = self.fc(x)

        

#         return logits



def loaderify(data, train):

    batch_size = 64 if train else 1

    shuffle = True

    num_workers = 2

    

    loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    

    return loader



def get_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):

    train_data = torch.from_numpy(X_train)

    val_data = torch.from_numpy(X_val)

    test_data = torch.from_numpy(X_test)

    

    train_labels = torch.tensor(Y_train)

    val_labels = torch.tensor(Y_val)

    test_labels = torch.tensor(Y_test)

    

    train_ds = torch.utils.data.TensorDataset(train_data, train_labels)

    val_ds = torch.utils.data.TensorDataset(val_data, val_labels)

    test_ds = torch.utils.data.TensorDataset(test_data, test_labels)

    

    train_loader = loaderify(train_ds, train = True)

    val_loader = loaderify(val_ds, train = False)

    test_loader = loaderify(test_ds, train = False)

    

    return [train_loader, val_loader, test_loader]



def fix_shape_neural(X_train, X_val, X_test):

    X_train_new = np.vstack(X_train)

    X_val_new = np.vstack(X_val)

    X_test_new = np.vstack(X_test)

    

    return [X_train_new, X_val_new, X_test_new]



def train_cnn(convnet, loader, val_loader, epochs = 1):

    criterion = nn.CrossEntropyLoss()

#     optimizer = optim.SGD(convnet.parameters(), lr = 0.01, momentum = 0.9)

    optimizer = optim.Adam(convnet.parameters(), lr = 0.0003)

    

    for epoch in range(epochs):

        running_loss = 0.0

        for i, data in enumerate(loader, 0):

            inputs, labels = data



            optimizer.zero_grad()



            outputs = convnet(inputs.cuda())

            loss = criterion(outputs, labels.cuda())

            loss.backward()

            optimizer.step()



            running_loss += loss.item()

            

            every_n = 100

            if i % every_n == every_n - 1:

                val_loss = get_val_loss(convnet, val_loader)

                

                print('[%d, %5d] loss: %.3f | val_loss: %.3f' % (epoch + 1, i + 1, running_loss / every_n, val_loss))

                running_loss = 0.0



def get_val_loss(convnet, val_loader):

    criterion = nn.CrossEntropyLoss()

    

    running_loss = 0.0

    for i, data in enumerate(val_loader, 0):

        inputs, labels = data



        outputs = convnet(inputs.cuda())

        loss = criterion(outputs, labels.cuda())



        running_loss += loss.item()

    

    return running_loss / len(val_loader)



def get_predictions(convnet, data_loader):

    y_true = []

    y_pred = []

    

    for data in data_loader:

        d, labels = data

        d = d.cuda()

        labels = labels.cuda()

        

        outputs = convnet(d)

        _, predicted = torch.max(outputs.data, 1)

        y_true += labels.tolist()

        y_pred += predicted.tolist()

    

    return [y_true, y_pred]



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):

    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length

    empty_cell = " " * columnwidth

    

    # Begin CHANGES

    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    

    if len(fst_empty_cell) < len(empty_cell):

        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell

    # Print header

    print("    " + fst_empty_cell, end=" ")

    # End CHANGES

    

    for label in labels:

        print("%{0}s".format(columnwidth) % label, end=" ")

        

    print()

    # Print rows

    for i, label1 in enumerate(labels):

        print("    %{0}s".format(columnwidth) % label1, end=" ")

        for j in range(len(labels)):

            cell = "%{0}.1f".format(columnwidth) % cm[i, j]

            if hide_zeroes:

                cell = cell if float(cm[i, j]) != 0 else empty_cell

            if hide_diagonal:

                cell = cell if i != j else empty_cell

            if hide_threshold:

                cell = cell if cm[i, j] > hide_threshold else empty_cell

            print(cell, end=" ")

        print()



def print_metrics(convnet, test_loader, classes):

    y_true, y_pred = get_predictions(convnet, test_loader)

    

    print('\nclassification report:')

    print(metrics.classification_report(y_true, y_pred, target_names = classes))

    

    y_true = [classes[a] for a in y_true]

    y_pred = [classes[a] for a in y_pred]

    

    cm = metrics.confusion_matrix(y_true, y_pred, labels = classes)

    print('\nconfusion matrix:')

    print_cm(cm, classes)



def evaluate(convnet, test_loader):

    y_true, y_pred = get_predictions(convnet, test_loader)

    

    correct = (np.array(y_true) == np.array(y_pred)).sum()

    total = len(y_true)

    evaluation = correct / total

    

    return evaluation



def preprocess_neural(data):

    data = remove_unlabeled(data)



    dev_data = data[data[:, 1] == 'train']

    test_data = data[data[:, 1] == 'test']

    train_data, val_data = train_test_split(dev_data, test_size = 0.05)

    

    X_train = train_data[:, 2]

    Y_train = train_data[:, 3]

    

    X_val = val_data[:, 2]

    Y_val = val_data[:, 3]

    

    X_test = test_data[:, 2]

    Y_test = test_data[:, 3]

    

    max_seq_len = 200

    max_vocab_size = 10000

    

    text2id = SimpleTextTransformer(max_vocab_size)

    

    print('Learning vocab...')

    

    X_train = text2id.fit_transform(X_train)

    X_train = np.array([pad(v, max_seq_len, text2id.pad_index) for v in X_train])

    

    X_val = text2id.transform(X_val)

    X_val = np.array([pad(v, max_seq_len, text2id.pad_index) for v in X_val])

    

    X_test = text2id.transform(X_test)

    X_test = np.array([pad(v, max_seq_len, text2id.pad_index) for v in X_test])

    

    Y_train = np.array([1 if Y_train[i] == 'pos' else 0 for i in range(Y_train.shape[0])])

    Y_val = np.array([1 if Y_val[i] == 'pos' else 0 for i in range(Y_val.shape[0])])

    Y_test = np.array([1 if Y_test[i] == 'pos' else 0 for i in range(Y_test.shape[0])])

    

    classes = ['neg', 'pos']

    vocab_size = len(text2id.vocab)

    

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test, classes, vocab_size]



def train_neural(train_loader, val_loader, test_loader, classes, vocab_size):

    emb_size = 300

    num_filters = 128

    num_classes = 2

    

    epochs = 5



    clf = TextClassifier(vocab_size, emb_size, num_filters, num_classes)

    clf.cuda()

    clf.train()

    

    print('Training...')

    

    train_cnn(clf, train_loader, val_loader, epochs)

    

    clf.eval()

    

#     val_loss = get_val_loss(clf, val_loader)

#     print('\nval_loss: {:.3f}\n'.format(val_loss))

    

    print_metrics(clf, test_loader, classes)

    

    score = evaluate(clf, val_loader)

    print('\nCNN score: {:.3f}\n'.format(score))

    

def main():

    unrandomize()

    

    data = get_data()

    X_train, Y_train, X_val, Y_val, X_test, Y_test, classes, vocab_size = preprocess_neural(data)

    train_loader, val_loader, test_loader = get_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    train_neural(train_loader, val_loader, test_loader, classes, vocab_size)



main()

print('done')