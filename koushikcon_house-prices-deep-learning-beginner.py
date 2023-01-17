import numpy as np

import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler



from sklearn.metrics import accuracy_score

import json

import os

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import SimpleImputer

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_boston

from sklearn.metrics import r2_score



import torch.utils.data

from sklearn.model_selection import train_test_split



import torch

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
os.mkdir('save')
train_csv = pd.read_csv("../input/train.csv", keep_default_na=False)

test_csv = pd.read_csv("../input/test.csv", keep_default_na=False)



# train = train[0:30]

print(train_csv.columns)
def preprocess_data(dataset):

    dataset['LotFrontage'] = pd.to_numeric(dataset["LotFrontage"].str.strip().replace("NA", ""))

    dataset['MasVnrArea'] = pd.to_numeric(dataset["MasVnrArea"].str.strip().replace("NA", ""))

    dataset['GarageYrBlt'] = pd.to_numeric(dataset["GarageYrBlt"].str.strip().replace("NA", ""))

    

    dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean(), inplace=True)

    dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean(), inplace=True)

    dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean(), inplace=True)

    

    dataset = dataset.drop(['Id'],axis=1)

    return dataset
train = preprocess_data(train_csv)

test = preprocess_data(test_csv)



display(train.head())
# train.dtypes
def combinelabelencode(cols, traindataset, testdataset):

    finallist = []

    for c in cols:

        finallist = np.concatenate((finallist, traindataset[c].values, testdataset[c].values), axis=None)

    

    finallist = np.unique(finallist)

    print(list(finallist))

    conditionlbl = LabelEncoder()

    conditionlbl.fit(list(finallist))

    for c in cols:

        traindataset[c] = conditionlbl.transform(list(traindataset[c].values))

        testdataset[c] = conditionlbl.transform(list(testdataset[c].values))

    return traindataset, testdataset
cols = ('MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig', 'LandSlope', 'Neighborhood',

        'RoofStyle', 'RoofMatl', 'SaleType', 'SaleCondition', 'MiscFeature', 'Fence', 'PoolQC', 'PavedDrive',

        'Functional', 'Electrical', 'Heating', 'Foundation', 'GarageType', 'BldgType', 'HouseStyle', 'CentralAir')

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(np.concatenate((train[c].values, test[c].values), axis=None))) 

    train[c] = lbl.transform(list(train[c].values))

    test[c] = lbl.transform(list(test[c].values))



train, test = combinelabelencode(('Condition1', 'Condition2'), train, test)

train, test = combinelabelencode(('BsmtQual', 'BsmtCond', 'BsmtExposure', 'KitchenQual', 'HeatingQC'), train, test)

train, test = combinelabelencode(('BsmtFinType1', 'BsmtFinType2', 'GarageFinish'), train, test)

train, test = combinelabelencode(('GarageQual', 'GarageCond', 'FireplaceQu', 'ExterQual', 'ExterCond'), train, test)

train, test = combinelabelencode(('Exterior1st', 'Exterior2nd', 'MasVnrType'), train, test)

display(train.head())
cat = len(train.select_dtypes(include=['object']).columns)

num = len(train.select_dtypes(include=['int64','float64']).columns)

print('Total Features: ', cat, 'categorical', '+', num, 'numerical', '=', cat+num, 'features')

print(train.select_dtypes(include=['object']).columns)
k = 52

min_val_corr = 0.4

corrmat = train.corr()

ser_corr = corrmat.nlargest(k, 'SalePrice')['SalePrice']

cols = ser_corr.index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=0.75)

f, ax = plt.subplots(figsize=(35, 35))

sns.heatmap(cm, cbar=True, annot=True, square=True, vmax=.8, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)

cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)

print(cols_abv_corr_limit)

print(cols_bel_corr_limit)
trimed_train = pd.DataFrame(columns=cols_abv_corr_limit)

trimed_test = pd.DataFrame(columns=cols_abv_corr_limit)

for c in cols_abv_corr_limit:

    trimed_train[c] = train[c]

    try:

        trimed_test[c] = test[c]

    except:

        print("column not found")



trimed_test = trimed_test.drop('SalePrice',axis = 1)



for c in trimed_test.select_dtypes(include=['object']).columns:

    trimed_test[c] = pd.to_numeric(trimed_test[c].str.strip().replace("NA", ""))

    trimed_test[c].fillna(trimed_test[c].median(), inplace=True)

# trimed_train = pd.concat([trimed_train, trimed_train], ignore_index=True)



col_train = list(cols_abv_corr_limit)

col_train_bis = list(cols_abv_corr_limit)



col_train_bis.remove('SalePrice')

print(len(col_train))

print(trimed_train.shape)



display(trimed_test.head())
mat_train = np.matrix(trimed_train)

mat_test  = np.matrix(trimed_test)



mat_new = np.matrix(trimed_train.drop('SalePrice',axis = 1))

mat_y = np.array(trimed_train.SalePrice).reshape((1460,1))



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y)



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



# trimed_test.to_csv("output_final_3.csv")

train_set = pd.DataFrame(prepro.transform(trimed_train),columns = col_train)



# test = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



test_set  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



display(train_set.head())
COLUMNS = col_train

FEATURES = col_train_bis

LABEL = "SalePrice"



# Training set and Prediction set with the features to predict

training_set = train_set[COLUMNS]

prediction_set = training_set.SalePrice



# print(prediction_set)



X_train, X_val, y_train, y_val = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.4)



train_set_tensor = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))

val_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values))



batch_size = 8

train_loader = torch.utils.data.DataLoader(train_set_tensor,batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)



# test_loader = torch.utils.data.DataLoader(torch.FloatTensor(test.values),batch_size=batch_size)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
# Hyperparameters

batch_no = len(X_train) // batch_size  #batches

cols=X_train.shape[1] #Number of columns in input matrix

n_output=1



# Sequence Length

sequence_length = 6  # of words in a sequence 892110

# Batch Size

# batch_size = 128

# train_loader = batch_data(int_text, sequence_length, batch_size)

# Number of Epochs

num_epochs = 1000

# Learning Rate

learning_rate = 0.002

# Model parameters

# Input size

input_size = cols

# Output size

output_size = 1

# Embedding Dimension

embedding_dim = 128

# Hidden Dimension

hidden_dim = 256

# Number of RNN Layers

n_layers = 2



# Show stats for every n number of batches

show_every_n_batches = 50
import torch.nn as nn



class LSTMClassifier(nn.Module):

    """

    This is the simple RNN model we will be using to perform Sentiment Analysis.

    """



    def __init__(self, embedding_dim, hidden_dim, input_size, n_layers, output_size, dropout=0.5):

        """

        Initialize the model by settingg up the various layers.

        """

        super(LSTMClassifier, self).__init__()



        # self.embedding = nn.Embedding(input_size, embedding_dim)

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.dense = nn.Linear(in_features=hidden_dim, out_features=output_size)

        self.sig = nn.Sigmoid()        

        # self.word_dict = None

        

        # self.fc1 = nn.Linear(input_size, hidden_dim * 2)

        # self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # self.fc3 = nn.Linear(hidden_dim, output_size)

        # self.dropout = nn.Dropout(p=0.25)

        self.init_weights()

        

    def init_weights(self):

        initrange = 0.08

        # self.embedding.weight.data.uniform_(-initrange, initrange)

        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)

        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)

        

        self.lstm.bias_ih_l0.data.zero_()

        self.lstm.bias_hh_l0.data.zero_()

        

        # self.fc.bias.data.zero_()

        self.dense.bias.data.fill_(0)

        # self.fc.weight.data.uniform_(-initrange, initrange)

        self.dense.weight.data.normal_(0.0, (1.0 / np.sqrt(self.dense.in_features)))

        

    def forward(self, x):

        """

        Perform a forward pass of our model on some input.

        """

        batch_size = x.size(0)

        # print(x.shape)

        # x = x.permute(14, 32)

        # x = F.relu(self.fc1(x))

        # x = self.dropout(x)

        # x = F.relu(self.fc2(x))

        # x = self.dropout(x)

        # out = self.fc3(x)

        # x = x.permute(2, 0, 1)

        # print(batch_size)

        # print(x.shape)

        # x = x.t()

        # embeds = self.embedding(x)

        lstm_out, _ = self.lstm(x)

        # avg_pool_l = torch.mean(lstm_out.permute(1, 0, 2), 1)

        # max_pool_l, _ = torch.max(lstm_out.permute(1, 0, 2), 1)

        # print(avg_pool_l)

        # x = torch.cat((avg_pool_l, max_pool_l), 1)

        # print(x.shape)

        out = self.dense(lstm_out)

        # out = self.sig(out.squeeze())

        # print(out)

        # out = out[lengths - 1, range(len(lengths))]

        return out
from torch.autograd import Variable



def forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden_dim, clip=9):



    if(train_on_gpu):

        inputs, labels = inputs.cuda(), labels.cuda()



    hidden = {}

    # hidden = tuple([each.data for each in hidden_dim])

    

    rnn.zero_grad()

    optimizer.zero_grad()

    

    try:

        # get the output from the model

        # output, hidden = rnn(inputs, hidden)

        output = rnn(inputs.unsqueeze(0))

        output = output.squeeze()

        # print(output.shape)

    except RuntimeError:

        raise

    # print(labels)

    loss = criterion(output, labels)

    loss.backward()

    

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

    nn.utils.clip_grad_norm_(rnn.parameters(),  clip)

   

    optimizer.step()



    return loss.item(), hidden
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):

    batch_losses = []

    val_batch_losses = []

    valid_loss_min = np.Inf

    

    rnn.train()

    

    previousLoss = np.Inf

    minLoss = np.Inf



    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):

        

        # initialize hidden state

        # hidden = rnn.init_hidden(batch_size)

        hidden = {}

        # print("epoch ",epoch_i)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # batch_last = batch_i

            # n_batches = len(train_loader.dataset) // batch_size

            

            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, clip=5)          

            # record loss

            batch_losses.append(loss)

            

        for batch_i, (inputs, labels) in enumerate(val_loader, 1):

            # batch_last = batch_i

            # n_batches = len(val_loader.dataset) // batch_size

            # if(batch_i > n_batches):

                # break

            

            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, clip=5)          

            # record loss

            val_batch_losses.append(loss)



        # printing loss stats

        if epoch_i%show_every_n_batches == 0:

            average_loss = np.average(batch_losses)

            val_average_loss = np.average(val_batch_losses)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, average_loss, val_average_loss))



            ## TODO: save the model if validation loss has decreased

            # save model if validation loss has decreased

            if val_average_loss <= valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                val_average_loss))

                with open('./save/trained_rnn_new', 'wb') as pickle_file:

                    # print(pickle_file)

                    torch.save(rnn, pickle_file)

                valid_loss_min = val_average_loss



            batch_losses = []

            val_batch_losses = []

            

    return rnn
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('No GPU found. Please use a GPU to train your neural network.')
# create model and move to gpu if available

# rnn = RNN(input_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.25)

# rnn.apply(weight_init)

rnn = LSTMClassifier(embedding_dim, hidden_dim, input_size, n_layers, output_size)

# rnn = torch.load("./save/trained_rnn_new")



if train_on_gpu:

    rnn.cuda()



decay_rate = learning_rate / num_epochs



# print(decay_rate)

# defining loss and optimization functions for training

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay_rate)



# criterion = nn.CrossEntropyLoss()

criterion = nn.MSELoss(size_average=False)

# rnn = helper.load_model('./save/trained_rnn_new')



# training the model

trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)



# saving the trained model

# helper.save_model('./save/trained_rnn', trained_rnn)

print('Model Trained and Saved')
def predict(model, inputs):



    if(train_on_gpu):

        inputs = inputs.cuda()

    

    try:

        output = model(inputs.unsqueeze(0))

        output = output.squeeze()

    except RuntimeError:

        raise

    

    # prediction = np.array(output).argmax(0)

    # p = F.softmax(output, dim=1).data

    # p = F.sigmoid(output)

    # p = F.logsigmoid(output)

    p = output.cpu().detach().numpy().flatten()

    # print(p)

    # prediction = np.argmax(p)

    # print(prediction)

    return p
model_rnn = torch.load("save/trained_rnn_new")

model_rnn.eval()



X = Variable(torch.FloatTensor(X_train.values)) 

pred = predict(model_rnn, X)

print(pred[:30])

# pred= result

print(y_train.values[:30])

r2_score(y_train.values, pred)



# probs = probs[:, 1]

# calculate AUC

# auc = roc_auc_score(y_train, pred)

# print('AUC: %.3f' % auc)

# calculate roc curve

# fpr, tpr, thresholds = roc_curve(y_train, pred)

# plot no skill

# pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

# pyplot.plot(fpr, tpr, marker='.')

# show the plot

# pyplot.show()

display(test_set.head())
test_X = Variable(torch.FloatTensor(test_set.values))

print(test_X)

test_pred = predict(model_rnn, test_X)

print(test_pred)

print(len(test_pred))

# print(np.array(test_p).reshape(9614,1))
predictions = pd.DataFrame(test_csv["Id"], columns = ["Id"])

predictions["SalePrice"] = prepro_y.inverse_transform(np.array(test_pred).reshape(1459,1))

# predictions = pd.DataFrame(np.array(test_pred).reshape(8037,1), columns = ["FORECLOSURE"])

# predictions["FORECLOSURE"] = predictions["FORECLOSURE"]

# predictions['SalePrice'] = predictions['SalePrice']

# predictions['FORECLOSURE'] = predictions['FORECLOSURE'].apply(lambda x: 0 if x < 0.01 else 1)

# predictions['FORECLOSURE'] = predictions['FORECLOSURE'].apply(lambda x: 1 if x > 0 else x)

# predictions = predictions.round(2)

# predictions["Id"] = test_csv["Id"]

display(predictions.head())
predictions.to_csv("submission.csv")