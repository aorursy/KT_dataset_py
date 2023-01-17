import warnings

warnings.filterwarnings('ignore')

import os

import random



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



SEED = 2020





def seed_everything(seed=2020):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)





seed_everything(SEED)
from torch import nn





class RMSELoss(nn.Module):

    def __init__(self, eps=1e-6):

        super().__init__()

        self.mse = nn.MSELoss()

        self.eps = eps



    def forward(self, yhat, y):

        loss = torch.sqrt(self.mse(yhat, y) + self.eps)

        return loss





class MCRMSELoss(nn.Module):

    def __init__(self, num_scored=3):

        super().__init__()

        self.rmse = RMSELoss()

        self.num_scored = num_scored



    def forward(self, yhat, y):

        score = 0

        for i in range(self.num_scored):

            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored



        return score
import pandas as pd





def load_json(path):

    return pd.read_json(path, lines=True)



df = load_json('/kaggle/input/stanford-covid-vaccine/train.json')

df_test = load_json('/kaggle/input/stanford-covid-vaccine/test.json')

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

df = df[df.SN_filter == 1]


df.head()
target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

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



train_inputs = torch.tensor(preprocess_inputs(df)).to(device)

print("input shape: ", train_inputs.shape)

train_labels = torch.tensor(

    np.array(df[target_cols].values.tolist()).transpose(0, 2, 1)

).float().to(device)
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

        self.linear = nn.Linear(hidden_dim * 2, len(target_cols))



    def forward(self, seqs):

        embed = self.embeding(seqs)

        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        output, hidden = self.gru(reshaped)

        truncated = output[:, : self.pred_len, :]

        out = self.linear(truncated)

        return out



criterion = MCRMSELoss(len(target_cols))



def compute_loss(batch_X, batch_Y, model, optimizer=None, is_train=True):

    model.train(is_train)



    pred_Y = model(batch_X)



    loss = criterion(pred_Y, batch_Y)



    if is_train:

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    return loss.item()
FOLDS = 4

EPOCHS = 90

BATCH_SIZE = 64

VERBOSE = 2

LR = 0.01
#get different test sets and process each

public_df = df_test.query("seq_length == 107").copy()

private_df = df_test.query("seq_length == 130").copy()



public_inputs = torch.tensor(preprocess_inputs(public_df)).to(device)

private_inputs = torch.tensor(preprocess_inputs(private_df)).to(device)



public_loader = DataLoader(TensorDataset(public_inputs), shuffle=False, batch_size=BATCH_SIZE)

private_loader = DataLoader(TensorDataset(private_inputs), shuffle=False, batch_size=BATCH_SIZE)
lstm_histories = []

lstm_private_preds = np.zeros((private_df.shape[0], 130, len(target_cols)))

lstm_public_preds = np.zeros((public_df.shape[0], 107, len(target_cols)))



criterion = MCRMSELoss()



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

        train_losses.append(np.mean(train_losses_batch))

        val_losses.append(np.mean(val_losses_batch))

    model_state = model.state_dict()

    del model

            

    lstm_histories.append({'train_loss': train_losses, 'val_loss': val_losses})





    lstm_short = LSTM_model(seq_len=107, pred_len=107).to(device)

    lstm_short.load_state_dict(model_state)

    lstm_short.eval()

    lstm_public_pred = np.ndarray((0, 107, len(target_cols)))

    for batch in public_loader:

        batch_X = batch[0]

        pred = lstm_short(batch_X).detach().cpu().numpy()

        lstm_public_pred = np.concatenate([lstm_public_pred, pred], axis=0)

    lstm_public_preds += lstm_public_pred / FOLDS



    lstm_long = LSTM_model(seq_len=130, pred_len=130).to(device)

    lstm_long.load_state_dict(model_state)

    lstm_long.eval()

    lstm_private_pred = np.ndarray((0, 130, len(target_cols)))

    for batch in private_loader:

        batch_X = batch[0]

        pred = lstm_long(batch_X).detach().cpu().numpy()

        lstm_private_pred = np.concatenate([lstm_private_pred, pred], axis=0)

    lstm_private_preds += lstm_private_pred / FOLDS

    

    del lstm_short, lstm_long


fig, ax = plt.subplots(1, 1, figsize = (20, 10))



for history in lstm_histories:

    ax.plot(history['train_loss'], 'b')

    ax.plot(history['val_loss'], 'r')



ax.set_title('LSTM')



ax.legend(['train', 'validation'], loc = 'upper right')



ax.set_ylabel('Loss')

ax.set_xlabel('Epoch');



public_df = df_test.query("seq_length == 107").copy()

private_df = df_test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
preds_lstm = []



for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_lstm.append(single_df)



preds_lstm_df = pd.concat(preds_lstm)

preds_lstm_df.head()
submission = sample_sub[['id_seqpos']].merge(preds_lstm_df, on=['id_seqpos'])
submission['deg_pH10'] = 0

submission['deg_50C'] = 0
submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')
