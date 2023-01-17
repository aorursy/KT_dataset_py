import numpy as np

import pandas as pd

import os

import torch

import torch.nn as nn

import time

import copy

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from torch.optim.lr_scheduler import ReduceLROnPlateau
n_epochs = 250

lr = 0.01

n_folds = 5

lstm_input_size = 1

hidden_state_size = 30

batch_size = 30

num_sequence_layers = 2

output_dim = 11

num_time_steps = 4000

rnn_type = 'LSTM'
class Bi_RNN(nn.Module):



    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, num_layers=2, rnn_type='LSTM'):

        super(Bi_RNN, self).__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.num_layers = num_layers



        #Define the initial linear hidden layer

        self.init_linear = nn.Linear(self.input_dim, self.input_dim)



        # Define the LSTM layer

        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)



        # Define the output layer

        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)



    def init_hidden(self):

        # This is what we'll initialise our hidden state as

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),

                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))



    def forward(self, input):

        #Forward pass through initial hidden layer

        linear_input = self.init_linear(input)



        # Forward pass through LSTM layer

        # shape of lstm_out: [batch_size, input_size ,hidden_dim]

        # shape of self.hidden: (a, b), where a and b both

        # have shape (batch_size, num_layers, hidden_dim).

        lstm_out, self.hidden = self.lstm(linear_input)



        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction

        y_pred = self.linear(lstm_out)

        return y_pred
class ION_Dataset_Sequential(Dataset):

    def __init__(self, input, output):

        self.input = input

        self.output = output



    def __len__(self):

        return len(self.input)



    def __getitem__(self, idx):

        x = self.input[idx]

        y = self.output[idx]

        x = torch.tensor(x, dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)

        return x, y



class ION_Dataset_Sequential_test(Dataset):

    def __init__(self, input):

        self.input = input



    def __len__(self):

        return len(self.input)



    def __getitem__(self, idx):

        x = self.input[idx]

        x = torch.tensor(x, dtype=torch.float)

        return x
train_df = pd.read_csv('/kaggle/input/data-no-drift/train_detrend.csv')

test_df = pd.read_csv('/kaggle/input/data-no-drift/test_detrend.csv')

X = train_df['signal'].values.reshape(-1, num_time_steps, 1)

y = pd.get_dummies(train_df['open_channels']).values.reshape(-1, num_time_steps, output_dim)

test_input = test_df["signal"].values.reshape(-1, num_time_steps, 1)

train_input_mean = X.mean()

train_input_sigma = X.std()

test_input = (test_input-train_input_mean)/train_input_sigma

test_preds = np.zeros((int(test_input.shape[0] * test_input.shape[1])))

test = ION_Dataset_Sequential_test(test_input)

test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
#Iterate through folds



kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

local_val_score = 0

models = {}



k=0 #initialize fold number

for tr_idx, val_idx in kfold.split(X, y):

    test_p = np.zeros((int(test_input.shape[0] * test_input.shape[1])))



    print('starting fold', k)

    k += 1



    print(6*'#', 'splitting and reshaping the data')

    train_input = X[tr_idx]

    print(train_input.shape)

    train_target = y[tr_idx]

    val_input = X[val_idx]

    val_target = y[val_idx]

    train_input_mean = train_input.mean()

    train_input_sigma = train_input.std()

    val_input = (val_input-train_input_mean)/train_input_sigma

    train_input = (train_input-train_input_mean)/train_input_sigma



    print(6*'#', 'Loading')

    train = ION_Dataset_Sequential(train_input, train_target)

    valid = ION_Dataset_Sequential(val_input, val_target)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)



    #Build tensor data for torch

    train_preds = np.zeros((int(train_input.shape[0] * train_input.shape[1])))

    val_preds = np.zeros((int(val_input.shape[0] * val_input.shape[1])))

    best_val_preds = np.zeros((int(val_input.shape[0] * val_input.shape[1])))

    train_targets = np.zeros((int(train_input.shape[0] * train_input.shape[1])))

    avg_losses_f = []

    avg_val_losses_f = []



    #Define loss function

    loss_fn = torch.nn.BCEWithLogitsLoss()



    #Build model, initialize weights and define optimizer

    model = Bi_RNN(lstm_input_size, hidden_state_size, batch_size=batch_size, output_dim=output_dim, num_layers=num_sequence_layers, rnn_type=rnn_type)  # (input_dim, hidden_state_size, batch_size, output_dim, num_seq_layers, rnn_type)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Using Adam optimizer

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.1, min_lr=1e-8)  # Using ReduceLROnPlateau schedule

    temp_val_loss = 9999999999

    reached_val_score = 0



    #Iterate through epochs

    for epoch in range(n_epochs):

        start_time = time.time()



        #Train

        model.train()

        avg_loss = 0.

        for i, (x_batch, y_batch) in enumerate(train_loader):

            x_batch = x_batch.view(-1, num_time_steps, lstm_input_size)

            y_batch = y_batch.view(-1, num_time_steps, output_dim)

            optimizer.zero_grad()

            y_pred = model(x_batch.cuda())

            loss = loss_fn(y_pred.cpu(), y_batch)

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

            pred = F.softmax(y_pred, 2).detach().cpu().numpy().argmax(axis=-1)

            train_preds[i * batch_size * train_input.shape[1]:(i + 1) * batch_size * train_input.shape[1]] = pred.reshape((-1))

            train_targets[i * batch_size * train_input.shape[1]:(i + 1) * batch_size * train_input.shape[1]] = y_batch.detach().cpu().numpy().argmax(axis=2).reshape((-1))

            del y_pred, loss, x_batch, y_batch, pred



        #Evaluate

        model.eval()

        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):

            x_batch = x_batch.view(-1, num_time_steps, lstm_input_size)

            y_batch = y_batch.view(-1, num_time_steps, output_dim)

            y_pred = model(x_batch.cuda()).detach()

            avg_val_loss += loss_fn(y_pred.cpu(), y_batch).item() / len(valid_loader)

            pred = F.softmax(y_pred, 2).detach().cpu().numpy().argmax(axis=-1)

            val_preds[i * batch_size * val_input.shape[1]:(i + 1) * batch_size * val_input.shape[1]] = pred.reshape((-1))

            del y_pred, x_batch, y_batch, pred

        if avg_val_loss < temp_val_loss:

            temp_val_loss = avg_val_loss



        #Calculate F1-score

        train_score = f1_score(train_targets, train_preds, average='macro')

        val_score = f1_score(val_target.argmax(axis=2).reshape((-1)), val_preds, average='macro')



        #Print output of epoch

        elapsed_time = time.time() - start_time

        scheduler.step(avg_val_loss)

        if epoch%10 == 0:

            print('Epoch {}/{} \t loss={:.4f} \t train_f1={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, train_score, avg_val_loss, val_score, elapsed_time))



        if val_score > reached_val_score:

            reached_val_score = val_score

            best_model = copy.deepcopy(model.state_dict())

            best_val_preds = copy.deepcopy(val_preds)



    #Calculate F1-score of the fold

    val_score_fold = f1_score(val_target.argmax(axis=2).reshape((-1)), best_val_preds, average='macro')



    #Save the fold's model in a dictionary

    models[k] = best_model



    #Print F1-score of the fold

    print("BEST VALIDATION SCORE (F1): ", val_score_fold)

    local_val_score += (1/n_folds) * val_score_fold



#Print final average k-fold CV F1-score

print("Final Score ", local_val_score)
#Iterate through folds



for k in range(n_folds):

    test_p = np.zeros((int(test_input.shape[0] * test_input.shape[1])))

    k += 1



    #Import model of fold k

    model = Bi_RNN(lstm_input_size, hidden_state_size, batch_size=batch_size, output_dim=output_dim, num_layers=num_sequence_layers, rnn_type=rnn_type)  # (input_dim, hidden_state_size, batch_size, output_dim, num_seq_layers, rnn_type)

    model = model.to(device)

    model.load_state_dict(models[k])



    #Make predictions on test data

    model.eval()

    for i, x_batch in enumerate(test_loader):

        x_batch = x_batch.view(-1, num_time_steps, lstm_input_size)

        y_pred = model(x_batch.cuda()).detach()

        pred = F.softmax(y_pred, 2).detach().cpu().numpy().argmax(axis=-1)

        test_p[i * batch_size * test_input.shape[1]:(i + 1) * batch_size * test_input.shape[1]] = pred.reshape((-1))

        del y_pred, x_batch, pred

    test_preds += (1/n_folds) * test_p
#Create submission file

df_sub = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv", dtype = {'time': str})

df_sub.open_channels = np.array(test_preds, np.int)

df_sub.to_csv("submission_bilstm.csv", index=False)