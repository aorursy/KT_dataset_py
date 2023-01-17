import torch

import logging

import argparse

import numpy as np

import pandas as pd

import torch.nn as nn

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from mne.time_frequency import psd_array_multitaper
def binary_acc(y_pred, y_test):



    # Apply sigmoid to obtain assigned probabilities

    # Apply rounding to convert it 0 or 1

    y_pred_tag = torch.round(torch.sigmoid(y_pred))



    # Number of correct predictions

    correct_results_sum = (y_pred_tag == y_test).sum().float()

    acc = correct_results_sum / y_test.shape[0]

    acc = torch.round(acc * 100)



    return acc

def get_target(df):

    return df.drop_duplicates('epoch')[['epoch', 'condition']].reset_index(drop=True)
def calc_features(df):

    feats = []

    ch_names = df.columns[3:]



    for epoch_idx, epoch_df in df.groupby('epoch'):



        epoch_df = epoch_df[ch_names]



        psds, freqs = psd_array_multitaper(epoch_df.T.values, 160, verbose=False)



        total_power = psds.sum(axis=1)



        idx_from = np.where(freqs > 13)[0][0]

        idx_to = np.where(freqs > 25)[0][0]



        d = {}

        d['epoch'] = epoch_idx



        for ch in ch_names:

            s = epoch_df.iloc[40:][ch]

            val = (s > 5).sum()

            d[ch.lower() + '_p300'] = val



        feats.append(d)



    feats_df = pd.DataFrame(feats)



    return feats_df

def train_net(net, device, epochs=100, lr=0.001, val_percent=0.1):

    df_train = pd.read_csv('/kaggle/input/neuroml2020eeg/train.csv')



    X = get_target(df_train)

    X = X.merge(calc_features(df_train), on='epoch')

    y = X['condition'].apply(lambda x: 0 if x == 1 else 1)



    del X['epoch']

    del X['condition']



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=val_percent, random_state=43)



    

    X = X.values

    y = y.values



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=val_percent, random_state=43)



    X_train_mean = X_train.mean(axis=0, keepdims=True)

    X_train_std = X_train.std(axis=0, keepdims=True)



    X_train_sc = (X_train - X_train_mean) / X_train_std

    X_test_sc = (X_test - X_train_mean) / X_train_std





    logging.info(f'''Starting training:

            Epochs:          {epochs}

            Learning rate:   {lr}

            Training size:   {X_train_sc.shape[0]}

            Validation size: {X_test_sc.shape[0]}

            Device:          {device.type}

        ''')



    x = torch.from_numpy(X_train_sc).float()

    y = torch.from_numpy(y_train).float()

    x_test = torch.from_numpy(X_test_sc).float()

    y_test = torch.from_numpy(y_test).float()





    loss_fn = nn.BCEWithLogitsLoss()



    optimizer = optim.Adam(net.parameters(), lr=lr)



    # assign large loss at the beginning

    min_loss = 1000

    best_acc = 0



    

    for e in range(epochs):

        net.train()

        y_pred = net(x)



        loss = loss_fn(y_pred, y.unsqueeze(dim=1))



        # Zero the gradients before running the backward pass.

        net.zero_grad()



        loss.backward()



        optimizer.step()



        if  ((e + 1) % 10) == 0:

            net.eval()

            with torch.no_grad():

                y_pred = net(x_test)

                loss = loss_fn(y_pred, y_test.unsqueeze(dim=1))

                print(e, "loss : ", loss.item())

                if loss < min_loss:

                    best_acc = binary_acc(y_pred, y_test.unsqueeze(dim=1))

                    print(e, "best acc : ", best_acc)

                    print(e, "best acc loss : ", loss.item())

                    min_loss = loss



    return min_loss, best_acc

def get_args():

    parser = argparse.ArgumentParser(description='Train neural network on training data')

    parser.add_argument('-s', '--size-of-hidden-layers', metavar='S', nargs=1, default=[8, 8],

                        help='Size of hidden layers', dest='size_of_hidden_layers')

    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5000,

                        help='Number of epochs', dest='epochs')

    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,

                        help='Learning rate', dest='lr')

    parser.add_argument('-v', '--validation', dest='val_per', type=float, default=30.0,

                        help='Percent of the data that is used as validation (0-100)')



    args, unknown = parser.parse_known_args()

    return args
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_in, D_out = 19, 1



    modules = [nn.Linear(D_in, args.size_of_hidden_layers[0]), nn.Sigmoid()]



    for i in range(1, len(args.size_of_hidden_layers)):

        modules.append(nn.Linear(args.size_of_hidden_layers[i - 1], args.size_of_hidden_layers[i]))

        modules.append(nn.Sigmoid())



    modules.append(

        nn.Linear(args.size_of_hidden_layers[-1], D_out)

    )



    model = nn.Sequential(*modules)



    model.to(device)



    min_loss, best_acc = train_net(net=model,

              epochs=args.epochs,

              lr=args.lr,

              device=device,

              val_percent=args.val_per / 100)



    logging.info(f'''Min Loss/Best Accuracy:

            Min Loss:          {min_loss}

            Best Acc:          {best_acc}

        ''')