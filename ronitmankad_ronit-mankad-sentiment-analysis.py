# General Imports

import copy

import os

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



# Pytorch Imports

import torch

from torch import nn

import torch.nn.functional as F

from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader



print(os.listdir('../input/'))
trainDF = pd.read_csv('../input/train.csv')

trainDF.head()
train_x, val_x, train_y, val_y = train_test_split(trainDF['review'], trainDF['sentiment'], test_size=0.1, shuffle=True)
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=25000)

tfidf_vect.fit(trainDF['review'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xval_tfidf = tfidf_vect.transform(val_x)
# Converting the Sparse matrix into a numpy array

xtrain_tfidf = xtrain_tfidf.toarray()

xval_tfidf = xval_tfidf.toarray()

# Converting pandas Series into numpy array

train_y = np.array(train_y)

val_y = np.array(val_y)
train_dataset = TensorDataset(torch.from_numpy(xtrain_tfidf).double(), torch.from_numpy(train_y).double())

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)



val_dataset = TensorDataset(torch.from_numpy(xval_tfidf).double(), torch.from_numpy(val_y).double())

val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)



dataloaders = {'train': train_dataloader, 'val': val_dataloader}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
# Hyper Parameters

epochs = 30

input_dim = 25000

output_dim = 1

lr_rate = 0.001
model = nn.Sequential(nn.Linear(input_dim, 2048),

                      nn.Dropout(0.5),

                      nn.ReLU(),

                      nn.Linear(2048, 256),

                      nn.Dropout(0.5),

                      nn.ReLU(),

                      nn.Linear(256, output_dim),

                      nn.Sigmoid()).double()
# Criterion and Optimizer

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
def train_model(model):

    model = model.cuda()

    best_acc = 0.0

    best_model_wts = copy.deepcopy(model.state_dict())

    

    for epoch in range(int(epochs)):

        train_loss = 0

        val_loss = 0

        val_acc = 0

        model.train()

        for inputs, labels in train_dataloader:

            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs.squeeze(), labels)

            loss.backward()

            optimizer.step()

            

            train_loss += loss.item()

        else:

            model.eval()

            num_correct = 0

            for inputs, labels in val_dataloader:

                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)

                predictions = torch.round(outputs.squeeze())

                loss = criterion(predictions, labels)

                

                val_loss += loss.item()

                equals = (predictions == labels.data)

    

                num_correct += torch.sum(equals.data).item()

            

            val_acc = num_correct / len(val_dataset)

            if val_acc > best_acc:

                best_acc = val_acc

                best_model_wts = copy.deepcopy(model.state_dict())

        print('---------Epoch {} -----------'.format(epoch))

        print('Train Loss: {:.6f} Val Loss: {:.6f} Val Accuracy: {:.6f}'.format(

                 train_loss/len(train_dataset), val_loss/len(val_dataset), val_acc))

        

    model.load_state_dict(best_model_wts)

    return model
model = train_model(model)
test_df = pd.read_csv('../input/test.csv')

xtest_tfidf =  tfidf_vect.transform(test_df['review'])

xtest_tfidf = xtest_tfidf.toarray()
test_y = np.zeros(xtest_tfidf.shape[0])
#predictions = classifier.predict(xtest_tfidf)

test_dataset = TensorDataset(torch.from_numpy(xtest_tfidf), torch.from_numpy(test_y))

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)
def predict(model, test_dataloader):

    model.eval()

    predictions = []

    for inputs, _ in test_dataloader:

        inputs = inputs.cuda()

        output = model(inputs)

        preds = torch.round(output)

        predictions.extend([p.item() for p in preds])

    return predictions



predictions = predict(model, test_dataloader)
sub_df = pd.DataFrame()

sub_df['Id'] = test_df['Id']

sub_df['sentiment'] = [int(p) for p in predictions]
sub_df.head()
sub_df.to_csv('my_submission.csv', index=False)
