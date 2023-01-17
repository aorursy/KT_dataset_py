import warnings

warnings.filterwarnings('ignore')

import os



#the basics

import pandas as pd, numpy as np, seaborn as sns

import math, json

from matplotlib import pyplot as plt

from tqdm import tqdm



#for model evaluation

from sklearn.model_selection import train_test_split, KFold



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#get comp data

train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
print(train.columns)
#sneak peak

print(train.shape)

if ~ train.isnull().values.any(): print('No missing values')

train.head()
#sneak peak

print(test.shape)

if ~ test.isnull().values.any(): print('No missing values')

test.head()
#sneak peak

print(sample_sub.shape)

if ~ sample_sub.isnull().values.any(): print('No missing values')

sample_sub.head()
fig, ax = plt.subplots(1, 2, figsize = (15, 5))

sns.kdeplot(train['signal_to_noise'], shade = True, ax = ax[0])

sns.countplot(train['SN_filter'], ax = ax[1])



ax[0].set_title('Signal/Noise Distribution')

ax[1].set_title('Signal/Noise Filter Distribution');
print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")

print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
#target columns

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )
train_filtered = train.loc[train.SN_filter == 1]

train_inputs = torch.tensor(preprocess_inputs(train_filtered)).to(device)

print("input shape: ", train_inputs.shape)

train_labels = torch.tensor(

    np.array(train_filtered[target_cols].values.tolist()).transpose(0, 2, 1)

).float().to(device)

print("output shape: ", train_labels.shape)
len(token2int)
class GRU_model(nn.Module):

    def __init__(

        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128, hidden_layers=3

    ):

        super(GRU_model, self).__init__()

        self.pred_len = pred_len



        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        self.gru = nn.GRU(

            input_size=embed_dim * 3,

            hidden_size=hidden_dim,

            num_layers=hidden_layers,

            dropout=dropout,

            bidirectional=True,

            batch_first=True,

        )

        self.linear = nn.Linear(hidden_dim * 2, 5)



    def forward(self, seqs):

        embed = self.embeding(seqs)

        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        output, hidden = self.gru(reshaped)

        truncated = output[:, : self.pred_len, :]

        out = self.linear(truncated)

        return out

    

class LSTM_model(nn.Module):

    def __init__(

        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128, hidden_layers=3

    ):

        super(LSTM_model, self).__init__()

        self.pred_len = pred_len



        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        self.gru = nn.LSTM(

            input_size=embed_dim * 3,

            hidden_size=hidden_dim,

            num_layers=hidden_layers,

            dropout=dropout,

            bidirectional=True,

            batch_first=True,

        )

        self.linear = nn.Linear(hidden_dim * 2, 5)



    def forward(self, seqs):

        embed = self.embeding(seqs)

        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        output, hidden = self.gru(reshaped)

        truncated = output[:, : self.pred_len, :]

        out = self.linear(truncated)

        return out

    

mse_loss = nn.MSELoss()

def compute_loss(batch_X, batch_Y, model, optimizer=None, is_train=True):

    model.train(is_train)



    pred_Y = model(batch_X)



    loss = mse_loss(pred_Y, batch_Y)



    if is_train:

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    return loss.item()
#basic training configuration

FOLDS = 4

EPOCHS = 90

BATCH_SIZE = 64

VERBOSE = 2

LR = 0.01
#get different test sets and process each

public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = torch.tensor(preprocess_inputs(public_df)).to(device)

private_inputs = torch.tensor(preprocess_inputs(private_df)).to(device)



public_loader = DataLoader(TensorDataset(public_inputs), shuffle=False, batch_size=BATCH_SIZE)

private_loader = DataLoader(TensorDataset(private_inputs), shuffle=False, batch_size=BATCH_SIZE)
gru_histories = []

gru_private_preds = np.zeros((private_df.shape[0], 130, 5))

gru_public_preds = np.zeros((public_df.shape[0], 107, 5))



kfold = KFold(FOLDS, shuffle=True, random_state=2020)



for k, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

    train_dataset = TensorDataset(train_inputs[train_index], train_labels[train_index])

    val_dataset = TensorDataset(train_inputs[val_index], train_labels[val_index])



    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)



    model = GRU_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)



    train_losses = []

    val_losses = []

    for epoch in tqdm(range(EPOCHS)):

        train_losses_batch = []

        val_losses_batch = []

        for (batch_X, batch_Y) in train_loader:

            train_loss = compute_loss(batch_X, batch_Y, model, optimizer=optimizer, is_train=True)

            train_losses_batch.append(train_loss)

        for (batch_X, batch_Y) in val_loader:

            val_loss = compute_loss(batch_X, batch_Y, model, optimizer=optimizer, is_train=False)

            val_losses_batch.append(val_loss)

        train_losses.append(sum(train_losses_batch) / len(train_losses_batch))

        val_losses.append(sum(val_losses_batch) / len(val_losses_batch))

    model_state = model.state_dict()

    del model

            

    gru_histories.append({'train_loss': train_losses, 'val_loss': val_losses})





    gru_short = GRU_model(seq_len=107, pred_len=107).to(device)

    gru_short.load_state_dict(model_state)

    gru_short.eval()

    gru_public_pred = np.ndarray((0, 107, 5))

    for batch in public_loader:

        batch_X = batch[0]

        pred = gru_short(batch_X).detach().cpu().numpy()

        gru_public_pred = np.concatenate([gru_public_pred, pred], axis=0)

    gru_public_preds += gru_public_pred / FOLDS



    gru_long = GRU_model(seq_len=130, pred_len=130).to(device)

    gru_long.load_state_dict(model_state)

    gru_long.eval()

    gru_private_pred = np.ndarray((0, 130, 5))

    for batch in private_loader:

        batch_X = batch[0]

        pred = gru_long(batch_X).detach().cpu().numpy()

        gru_private_pred = np.concatenate([gru_private_pred, pred], axis=0)

    gru_private_preds += gru_private_pred / FOLDS

    

    del gru_short, gru_long
print(f" GRU mean fold validation loss: {np.mean([min(history['val_loss']) for history in gru_histories])}")
lstm_histories = []

lstm_private_preds = np.zeros((private_df.shape[0], 130, 5))

lstm_public_preds = np.zeros((public_df.shape[0], 107, 5))



kfold = KFold(FOLDS, shuffle=True, random_state=2020)



for k, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

    train_dataset = TensorDataset(train_inputs[train_index], train_labels[train_index])

    val_dataset = TensorDataset(train_inputs[val_index], train_labels[val_index])



    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)



    model = LSTM_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)



    train_losses = []

    val_losses = []

    for epoch in tqdm(range(EPOCHS)):

        train_losses_batch = []

        val_losses_batch = []

        for (batch_X, batch_Y) in train_loader:

            train_loss = compute_loss(batch_X, batch_Y, model, optimizer=optimizer, is_train=True)

            train_losses_batch.append(train_loss)

        for (batch_X, batch_Y) in val_loader:

            val_loss = compute_loss(batch_X, batch_Y, model, optimizer=optimizer, is_train=False)

            val_losses_batch.append(val_loss)

        train_losses.append(sum(train_losses_batch) / len(train_losses_batch))

        val_losses.append(sum(val_losses_batch) / len(val_losses_batch))

    model_state = model.state_dict()

    del model

            

    lstm_histories.append({'train_loss': train_losses, 'val_loss': val_losses})





    lstm_short = LSTM_model(seq_len=107, pred_len=107).to(device)

    lstm_short.load_state_dict(model_state)

    lstm_short.eval()

    lstm_public_pred = np.ndarray((0, 107, 5))

    for batch in public_loader:

        batch_X = batch[0]

        pred = lstm_short(batch_X).detach().cpu().numpy()

        lstm_public_pred = np.concatenate([lstm_public_pred, pred], axis=0)

    lstm_public_preds += lstm_public_pred / FOLDS



    lstm_long = LSTM_model(seq_len=130, pred_len=130).to(device)

    lstm_long.load_state_dict(model_state)

    lstm_long.eval()

    lstm_private_pred = np.ndarray((0, 130, 5))

    for batch in private_loader:

        batch_X = batch[0]

        pred = lstm_long(batch_X).detach().cpu().numpy()

        lstm_private_pred = np.concatenate([lstm_private_pred, pred], axis=0)

    lstm_private_preds += lstm_private_pred / FOLDS

    

    del lstm_short, lstm_long
print(f" LSTM mean fold validation loss: {np.mean([min(history['val_loss']) for history in lstm_histories])}")
fig, ax = plt.subplots(1, 2, figsize = (20, 10))



for history in gru_histories:

    ax[0].plot(history['train_loss'], 'b')

    ax[0].plot(history['val_loss'], 'r')

for history in lstm_histories:

    ax[1].plot(history['train_loss'], 'b')

    ax[1].plot(history['val_loss'], 'r')



ax[0].set_title('GRU')

ax[1].set_title('LSTM')



ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[1].legend(['train', 'validation'], loc = 'upper right')



ax[0].set_ylabel('Loss')

ax[0].set_xlabel('Epoch')

ax[1].set_ylabel('Loss')

ax[1].set_xlabel('Epoch');
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
preds_gru = []



for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_gru.append(single_df)



preds_gru_df = pd.concat(preds_gru)

preds_gru_df.head()
preds_lstm = []



for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_lstm.append(single_df)



preds_lstm_df = pd.concat(preds_lstm)

preds_lstm_df.head()
blend_preds_df = pd.DataFrame()

blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']

blend_preds_df['reactivity'] = .5*preds_gru_df['reactivity'] + .5*preds_lstm_df['reactivity']

blend_preds_df['deg_Mg_pH10'] = .5*preds_gru_df['deg_Mg_pH10'] + .5*preds_lstm_df['deg_Mg_pH10']

blend_preds_df['deg_pH10'] = .5*preds_gru_df['deg_pH10'] + .5*preds_lstm_df['deg_pH10']

blend_preds_df['deg_Mg_50C'] = .5*preds_gru_df['deg_Mg_50C'] + .5*preds_lstm_df['deg_Mg_50C']

blend_preds_df['deg_50C'] = .5*preds_gru_df['deg_50C'] + .5*preds_lstm_df['deg_50C']
submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])



#sanity check

submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')