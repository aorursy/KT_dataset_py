import numpy as np

import pandas as pd

import torch as tr
imdb = pd.read_csv('../input/imdb_master.csv', encoding='ISO-8859-1')

dev_df = imdb[(imdb.type == 'train') & (imdb.label != 'unsup')]



test_df = imdb[(imdb.type == 'test')]
from sklearn import model_selection

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.1, stratify=dev_df.label)
import re

from nltk.corpus import stopwords 

    

def word_extraction(sentence):   

    ignore = stopwords.words('english')

    words = re.sub("[^\w]", " ", sentence).split()    

    cleaned_text = [w.lower() for w in words if w not in ignore and not w.isdigit()]   

    return cleaned_text
def build_vocab(sentences):    

    for i, sentence in enumerate(sentences):

        words = word_extraction(sentence)

        for w in words:

            if w in vocab:

                vocab[w] = vocab[w]+1

            else:

                vocab[w] = 1

            

    return vocab
def get_max_seq_len(train_data, val_data, test_data):

    maxl = 0

    for sentence in train_data:  

        words = word_extraction(sentence)

        if maxl<len(words):

            maxl = len(words)

            

    for sentence in val_data:  

        words = word_extraction(sentence)

        if maxl<len(words):

            maxl = len(words)

            

    for sentence in test_data:  

        words = word_extraction(sentence)

        if maxl<len(words):

            maxl = len(words)

    

    return maxl
def generate_bow(t_vocab, allsentences):

    vocab_vector = list()

    size = len(allsentences)

    for cur_i, sentence in enumerate(allsentences):  

        if cur_i % 2000==0:

            print(cur_i,"/",size)

        cur_i += 1

        words = word_extraction(sentence)

        #print(words)

        bag_vector = np.full((2000), 0)

        for i, word in enumerate(words):

            try:

                v_i = vocab.index(word)

            except:

                v_i = 0

            #print(word,"-",int(v_i), "-", i)

            bag_vector[i] = v_i

        #print(bag_vector.tolist())

        vocab_vector.append(bag_vector)

    return vocab_vector
vocab = {".":1}

vocab_dic = build_vocab(train_df["review"])

print("Vocabulary with words count length ", len(vocab_dic.keys()))

#print(vocab_dic)



delete = [key for key, value in vocab_dic.items() if value < 80 and key!='.'] 

for key in delete: del vocab_dic[key] 



vocab = [key for key in vocab_dic.keys()]

print("Vocabulary length ", len(vocab))

#print(vocab)
print("train_data")

train_data = generate_bow(vocab, train_df["review"])

print("val_data")

val_data = generate_bow(vocab, val_df["review"])

print("test_data")

test_data = generate_bow(vocab, test_df["review"])
from torch.utils.data import TensorDataset,DataLoader

from torch.utils import data



def build_features(token_ids, label, max_seq_len, pad_index, label_encoding):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [0 for _ in range(max_seq_len - len(token_ids))]

    return [ids, [i for i,el in enumerate(label_encoding) if el == label][0]]



def features_to_tensor(list_of_features):

    text_tensor = tr.tensor([example[0] for example in list_of_features], dtype=tr.long)

    labels_tensor = tr.tensor([example[1] for example in list_of_features], dtype=tr.long)

    return text_tensor, labels_tensor

    

classes = ['neg', 'pos']



max_seq_len = get_max_seq_len(train_df["review"], val_df["review"], test_df["review"])



train_features = [build_features(review.tolist(), label, max_seq_len, 0, classes) for review, label in zip(train_data, train_df['label'])]

val_features = [build_features(review.tolist(), label, max_seq_len, 0, classes) for review, label in zip(val_data, val_df['label'])]

test_features = [build_features(review.tolist(), label, max_seq_len, 0, classes) for review, label in zip(test_data, test_df['label'])]



train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)



batch_size = 128

train_loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size = batch_size)

val_loader = DataLoader(TensorDataset(val_tensor, val_labels), batch_size = batch_size)

test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size = batch_size, shuffle=True)
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data

from sklearn import metrics



class TextClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, num_filters, num_classes):

        super(TextClassifier, self).__init__()



        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.dropout_big = tr.nn.Dropout(p = 0.4)

        self.dropout_small = tr.nn.Dropout(p = 0.05)



        self.convs = nn.ModuleList([

            nn.Conv1d(1, num_filters, [3, emb_size], padding = (2, 0)),

            nn.Conv1d(1, num_filters, [4, emb_size], padding = (3, 0)),

            nn.Conv1d(1, num_filters, [5, emb_size], padding = (4, 0)),

        ])



        self.fc = nn.Linear(num_filters * 3, num_classes)



    def forward(self, x):

        x = self.embedding(x)

        x = self.dropout_big(x)



        x = tr.unsqueeze(x, 1)

        xs = []

        for i in range(len(self.convs)):

            conv = self.convs[i]

            

            x2 = F.relu(conv(x))

            x2 = tr.squeeze(x2, -1)

            x2 = F.max_pool1d(x2, x2.size(2))

            

            if i == 1:

                x2 = self.dropout_small(x2)

            

            xs.append(x2)

        x = tr.cat(xs, 2)



        x = x.view(x.size(0), -1)

        logits = self.fc(x)

        

        return logits
def train(conv, loader, val_loader, epochs):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(conv.parameters(), lr = 0.001)



    for epoch in range(epochs):

        r_loss = 0.0

        for i, data in enumerate(loader, 0):

            inputs, labels = data



            optimizer.zero_grad()



            outputs = conv(inputs.cuda())

            loss = criterion(outputs, labels.cuda())

            loss.backward()

            optimizer.step()



            r_loss += loss.item()

            

            every_n = 100

            if i % every_n == every_n - 1:

                validation_loss = get_validation_loss(conv, val_loader)

                print('Epoch %d: loss: %.3f, validation loss: %.3f' %

                      (epoch + 1, r_loss / every_n, validation_loss))

                r_loss = 0.0
def get_validation_loss(conv, val_loader):

    criterion = nn.CrossEntropyLoss()

    

    validation_loss = 0.0

    for i, data in enumerate(val_loader, 0):

        inputs, labels = data



        outputs = conv(inputs.cuda())

        loss = criterion(outputs, labels.cuda())



        validation_loss += loss.item()

    

    return validation_loss / len(val_loader)
def test_predictions(conv, data_loader):

    y_true = []

    y_pred = []

    

    for data in data_loader:

        d, labels = data

        d = d.cuda()

        labels = labels.cuda()

        

        outputs = conv(d)

        _, predicted = tr.max(outputs.data, 1)

        y_true += labels.tolist()

        y_pred += predicted.tolist()

    

    return [y_true, y_pred]

def get_metrics(conv, test_loader, classes):

    y_true, y_pred = test_predictions(conv, test_loader)

    

    classification = metrics.classification_report(y_true, y_pred, target_names = classes)

    

    y_true = [classes[a] for a in y_true]

    y_pred = [classes[a] for a in y_pred]

    

    confusion = metrics.confusion_matrix(y_true, y_pred, labels = classes)

    return [classification, confusion]
def validate(conv, test_loader):

    y_true, y_pred = test_predictions(conv, test_loader)

    

    correct = (np.array(y_true) == np.array(y_pred)).sum()

    total = len(y_true)

    validation = correct / total

    

    return validation
def test(conv, test_loader):

    dataiter = iter(test_loader)



    for i in range(10):

        review, labels = next(dataiter)

        outputs = conv(review.cuda())

        _, predicteds = tr.max(outputs, 1)

        predicted = classes[predicteds[0]]

        groundtruth = classes[labels[0]]

        

        words = []

        for x in review[0]:

            if vocab[x]!=".":

                words.append(vocab[x])

            

        print()

        print(words)

        print("Prediction: {}".format(predicted))

        print("Real: {}".format(groundtruth))
emb_size = 100

num_filters = 500

num_classes = 2



len_vocab = len(vocab)

tc = TextClassifier(len_vocab, emb_size, num_filters, num_classes)

tc.cuda()



print('Train')

train(tc, train_loader, val_loader, 5)
classification, confusion = get_metrics(tc, test_loader, classes)



print()

print('Classification report:')

print(classification)



print()

print('Confusion report:')

print(confusion)
print()

print("Test")

test(tc, test_loader)
tc_validation = validate(tc, val_loader)

print('Validation score')

print(tc_validation)