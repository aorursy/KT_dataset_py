import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import random

import os

import math

import time

from torch.utils.data import DataLoader, Dataset
class SequenceLoader(Dataset):



    def __init__(self, dataframe_path, signal_noise_cutoff, test_set=None):

        super().__init__()

        self.df = pd.read_json(dataframe_path)

        deg_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

        

        self.is_test = test_set is not None or deg_cols[0] not in self.df.columns

        if self.is_test:

            self.df = self.df.query(("seq_length == 107" if test_set == 'public' else "seq_length == 130"))

            self.y = None

        else:

            self.df = self.df[self.df.signal_to_noise >= signal_noise_cutoff]

            self.y = np.stack([np.stack(self.df[col].values) for col in deg_cols], axis=-1)



        self.sample_ids = self.df['id'].values

        self.X = np.stack(self.df['train_tensor'].values)

        self.id_to_bp_mat_map = self.load_bp_mats()



    def __getitem__(self, index: int):

        x = torch.tensor(self.X[index, :, :], dtype=torch.float32)

        seq_adj = self.get_sequence_adjacency(x.size()[0])

        bp_adj = self.get_base_pair_adjacency(self.sample_ids[index])



        if self.is_test:

            sample_id = self.sample_ids[index]

            return sample_id, x, seq_adj, bp_adj



        targets = torch.tensor(self.y[index, :, :], dtype=torch.float32)

        return x, targets, seq_adj, bp_adj



    @staticmethod

    def get_sequence_adjacency(size):

        r_shift = np.pad(np.identity(size), ((0, 0), (1, 0)), mode='constant')[:, :-1]

        l_shift = np.pad(np.identity(size), ((0, 0), (0, 1)), mode='constant')[:, 1:]

        return torch.tensor(r_shift + l_shift, dtype=torch.float32)



    def get_base_pair_adjacency(self, sample_id):

        return self.id_to_bp_mat_map[sample_id]



    def load_bp_mats(self):

        res = {}

        for sid in self.sample_ids:

            res[sid] = torch.tensor(np.load('../input/stanford-covid-vaccine/bpps/' + sid + '.npy'), dtype=torch.float32)

        return res

    

    def __len__(self) -> int:

        return self.df.shape[0]





def dataset_loader(df_path, batch_size, signal_noise_cutoff=-99.0, test_set=None):

    dataset = SequenceLoader(df_path, signal_noise_cutoff, test_set=test_set)

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=(test_set is None),

        num_workers=4

    )
# Generates an encoding to capture a node's position (just like transformers).

# Used as part of the GraphBlock.

class PositionalEncoding(nn.Module):



    def __init__(self, d_model, dropout=0.2, max_len=300):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)



        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)



    def forward(self, x, offset=None):

        x = x + self.pe[:x.size(1), :]

        return self.dropout(x)

    



# Provides info about the entire graph to every node.

class GraphBlock(nn.Module):



    def __init__(self, mlp_size, dropout=0.2):

        super(GraphBlock, self).__init__()

        self.pos_encoder = PositionalEncoding(mlp_size)

        self.graph_layer = nn.Linear(mlp_size, mlp_size)

        self.layer_norm = torch.nn.LayerNorm(mlp_size)

        self.dropout = torch.nn.Dropout(dropout)



    def forward(self, x):

        graph_node_emb = torch.tanh(self.graph_layer(x))

        graph_node_emb = self.dropout(graph_node_emb)

        graph_emb = torch.mean(graph_node_emb, 1)

        graph_emb = graph_emb.reshape((graph_emb.size()[0], 1, graph_emb.size()[1]))



        pos_enc = self.pos_encoder(x)

        return self.dropout(self.layer_norm(pos_enc + graph_emb))





class ConvAttnBlock(nn.Module):



    def __init__(self, mlp_size, adj_conv_channels=4, dropout=0.2):

        super(ConvAttnBlock, self).__init__()

        self.adj_conv_channels = adj_conv_channels

        self.neighbor_layer = nn.Linear(mlp_size, mlp_size)

        self.result_layer = nn.Linear(mlp_size, mlp_size)

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=adj_conv_channels, 

                                     kernel_size=(7), padding=3)

        self.conv2 = torch.nn.Conv2d(in_channels=adj_conv_channels, 

                                     out_channels=adj_conv_channels, kernel_size=(17), padding=8)

        self.layer_norm = torch.nn.LayerNorm(mlp_size)

        self.dropout = torch.nn.Dropout(dropout)



    def forward(self, x, bp_adj):

        batch = bp_adj.size(0)

        d1 = bp_adj.size(1)

        d2 = bp_adj.size(2)



        bp_adj = self.conv1(bp_adj.reshape((batch, 1, d1, d2)))

        bp_adj = self.conv2(bp_adj)

        bp_adj = torch.mean(bp_adj, 1)[:, :d1, :d2]

        neighbor_emb = torch.tanh(self.neighbor_layer(x))

        neighbor_emb = self.dropout(neighbor_emb)

        neighbor_sum = torch.matmul(bp_adj, neighbor_emb)

        cat = x + neighbor_sum

        out = torch.tanh(self.result_layer(cat))

        out = self.layer_norm(out)

        return self.dropout(out), bp_adj





# Combines 2 rounds of conv attention on the sequence and BPP matrices.

# Includes data from 1 graph block and a skip connection.

class NeighborAttnStage(nn.Module):

    def __init__(self, mlp_size, dropout=0.2):

        super(NeighborAttnStage, self).__init__()

        self.graph_block = GraphBlock(mlp_size)

        self.sequence_block1 = ConvAttnBlock(mlp_size)

        self.sequence_block2 = ConvAttnBlock(mlp_size)

        self.base_pair_block1 = ConvAttnBlock(mlp_size)

        self.base_pair_block2 = ConvAttnBlock(mlp_size)

        

    def forward(self, x_in, seq_adj, bp_adj):

        x_bp1, _ = self.base_pair_block1(x_in, bp_adj)

        x_seq1, _ = self.sequence_block1(x_in, seq_adj)

        x = self.graph_block(x_in) + x_bp1 + x_seq1



        x_bp2, _ = self.base_pair_block2(x, bp_adj)

        x_seq2, _ = self.sequence_block2(x, seq_adj)

        return x_bp2 + x_seq2 + x_in





class NeighborhoodAttentionModel(nn.Module):



    def __init__(self, mlp_size, dropout=0.2):

        super(NeighborhoodAttentionModel, self).__init__()

        self.input_fc = nn.Linear(14, mlp_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv_attn_stage1 = NeighborAttnStage(mlp_size)

        self.conv_attn_stage2 = NeighborAttnStage(mlp_size)

        self.conv_attn_stage3 = NeighborAttnStage(mlp_size)

        self.output_fc1 = nn.Linear(mlp_size, mlp_size)

        self.output_fc2 = nn.Linear(mlp_size, 3)



    def forward(self, x, seq_adj, bp_adj):

        x = torch.tanh(self.input_fc(x))

        x = self.dropout(x)



        x = self.conv_attn_stage1(x, seq_adj, bp_adj)

        x = self.conv_attn_stage2(x, seq_adj, bp_adj)

        x = self.conv_attn_stage3(x, seq_adj, bp_adj)



        x = self.dropout(x)

        x = torch.tanh(self.output_fc1(x))

        

        x = self.dropout(x)

        return self.output_fc2(x)
# Uninteresting code used to keep track of average loss over an epoch

class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0

        self.start_time = time.time()



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    @property

    def time(self):

        return time.time() - self.start_time



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0

        self.start_time = time.time()





# Pytorch MCRMSE Losss

# [Link to Kernel]

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



# Get Device (CPU or GPU)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(device)



# Hyperparameters

epochs = 100

batch_size = 8

node_emb_size = 512

lr = 0.0001

lr_drop_epochs = 45

lr_gamma = 0.1

criterion = MCRMSELoss()



# Setup Train/Val Data Loaders

prepd_train_data = '../input/covidtrainvaldataset/train_1.json'

prepd_val_data = '../input/covidtrainvaldataset/val_1.json'

train_loader = dataset_loader(prepd_train_data, batch_size=batch_size, signal_noise_cutoff=0.6)

val_loader = dataset_loader(prepd_val_data, batch_size=batch_size, signal_noise_cutoff=1.0)



# Model

model = NeighborhoodAttentionModel(node_emb_size).to(device)



# Optimizer & Scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop_epochs, gamma=lr_gamma)



# Objects for keeping track of loss over epochs.

epoch_loss_hist = Averager()

val_loss_hist = Averager()





for epoch in range(epochs):

    model.train()

    epoch_loss_hist.reset()

    

    # Training

    for sequences, targets, seq_adj_matrix, bp_adj_matrix in train_loader:

        optimizer.zero_grad()

        pred = model(sequences.to(device), seq_adj_matrix.to(device), bp_adj_matrix.to(device))

        loss = criterion(pred[:, :targets.size()[1], :], targets.to(device))

        loss.backward()

        optimizer.step()

        epoch_loss_hist.send(loss.item())

        

    # Validation

    with torch.no_grad():

        model.eval()

        val_loss_hist.reset()



        for sequences, targets, seq_adj_matrix, bp_adj_matrix in val_loader:

            pred = model(sequences.to(device), seq_adj_matrix.to(device), bp_adj_matrix.to(device))

            loss = criterion(pred[:, :targets.size(1), :], targets.to(device))

            val_loss_hist.send(loss.item())



    print('Epoch:', epoch, 'Train Loss:', epoch_loss_hist.value, 'CV Loss:', val_loss_hist.value)

    scheduler.step()

def build_submission_df(ids, pred_tensor):

    if type(pred_tensor).__module__ != np.__name__:

        pred_tensor = pred_tensor.cpu().detach().numpy()

    res = []

    for i, id in enumerate(ids):

        for j, pred in enumerate(pred_tensor[i, :, :]):

            res.append([id+'_'+str(j)] + list(pred))

    return res





def make_pred_file(model, loaders, postfix=''):

    res = []

    for loader in loaders:

        for ids, sequences, seq_adj_matrix, bp_adj_matrix in loader:

            test_pred = model(sequences.to(device), seq_adj_matrix.to(device), bp_adj_matrix.to(device))

            res += build_submission_df(ids, test_pred)



    pred_df = pd.DataFrame(res, columns=['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])

    pred_df['deg_pH10'] = 0

    pred_df['deg_50C'] = 0

    pred_df.to_csv('./submission'+postfix+'.csv', index=False)

    





test_data_path = '../input/covidtrainvaldataset/test_1.json'

test_data_loader1 = dataset_loader(test_data_path, test_set='public', batch_size=batch_size)

test_data_loader2 = dataset_loader(test_data_path, test_set='private', batch_size=batch_size)

make_pred_file(model, [test_data_loader1, test_data_loader2])