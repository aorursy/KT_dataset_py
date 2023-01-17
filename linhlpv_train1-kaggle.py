import torch

from torch import nn

from torch.nn import Parameter

import torch.nn.functional as F



class GATConvLayer_new(nn.Module):

    def __init__(self,

                 node_in_fts,

                 node_out_fts,

                 ed_in_fts,

                 ed_out_fts,

                 state_dim,

                 concat=True,

                 negative_slope=0.1,

                 dropout=0.1,

                 bias=True,

                 repeat_edge = 1

                 ):

        super(GATConvLayer_new, self).__init__()

 

        self.node_in_fts = node_in_fts

        self.node_out_fts = node_out_fts

        self.ed_out_fts = ed_out_fts

        self.ed_in_fts = ed_in_fts

        self.concat = concat

        self.negative_slope = negative_slope

        self.dropout = dropout

        self.repeat_edge = repeat_edge

        self.state_dim = state_dim

 

        self.node_update_layer = nn.Linear(self.node_in_fts, node_out_fts)

        self.ed_trans_layer = nn.Linear(2 * self.node_in_fts + self.ed_in_fts, 2 * self.state_dim)

        self.ed_hidden_1 = nn.Linear(2 * self.state_dim, 2 * state_dim)

        self.ed_hidden_2 = nn.Linear(2 * self.state_dim, state_dim)

        self.ed_out_layer = nn.Linear(self.state_dim, self.ed_out_fts)

        self.e = Parameter(torch.zeros(size=(2 * self.node_out_fts + self.ed_out_fts, 1)))



        self.batch_norm_n = nn.BatchNorm1d(self.node_out_fts)

        self.batch_norm_e = nn.BatchNorm1d(self.ed_out_fts)

        

        if bias and concat:

            self.bias = Parameter(torch.Tensor(node_out_fts))

        elif bias and not concat:

            self.bias = Parameter(torch.Tensor(node_out_fts))

        else:

            self.register_parameter('bias', None)

 

        self.reset_parameters()

 

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.e.data, gain=1.414)

 

    def forward(self, x, edge_attr, mask):

        return self.propagate( x=x, num_nodes=x.size(0), edge_attr=edge_attr, mask = mask)

 

    def propagate(self, **kwargs):

        edge_attr = kwargs['edge_attr']

        x = kwargs['x']

        N = kwargs['num_nodes']

        mask = kwargs['mask']



        triplet_e = torch.cat([x.repeat(1, self.repeat_edge*N).view(self.repeat_edge*N*N, -1), edge_attr, x.repeat(self.repeat_edge*N, 1)], dim=-1)

        new_edge_attr = F.leaky_relu(self.ed_trans_layer(triplet_e), self.negative_slope)

        new_edge_attr = F.leaky_relu(self.ed_hidden_1(triplet_e), self.negative_slope)

        new_edge_attr = F.leaky_relu(self.ed_hidden_2(triplet_e), self.negative_slope)

        new_edge_attr = F.leaky_relu(self.ed_out_layer(new_edge_attr), self.negative_slope)

        new_edge_attr = new_edge_attr * mask.repeat(1, new_edge_attr.shape[1])



        x = F.leaky_relu(self.node_update_layer(x), self.negative_slope)

        triplet_n = torch.cat([x.repeat(1, self.repeat_edge*N).view(self.repeat_edge*N*N, -1), new_edge_attr, x.repeat(self.repeat_edge*N, 1)], dim=-1)

        triplet_n = triplet_n.view(N, -1, 2 * self.node_out_fts + self.ed_out_fts)

        e = F.leaky_relu(torch.matmul(triplet_n, self.e).squeeze(2), self.negative_slope)

        attention = F.softmax(e, dim = 1)

        attention = attention.view(N * N, -1)

        attention = attention * mask

        attention = attention.view(N, N)

        attention = F.dropout(attention, self.dropout, training=self.training)

        new_x = torch.matmul(attention, x)



        new_x = self.batch_norm_n(new_x)

        new_edge_attr = self.batch_norm_e(new_edge_attr)



        return [new_x, new_edge_attr]

import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)



                p.data.copy_(p_data_fp32)



        return loss



class PlainRAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)



        super(PlainRAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(PlainRAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                beta2_t = beta2 ** state['step']

                N_sma_max = 2 / (1 - beta2) - 1

                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:                    

                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                else:

                    step_size = group['lr'] / (1 - beta1 ** state['step'])

                    p_data_fp32.add_(-step_size, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss





class AdamW(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay, warmup = warmup)

        super(AdamW, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(AdamW, self).__setstate__(state)



    def step(self, closure=None):

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                

                if group['warmup'] > state['step']:

                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']

                else:

                    scheduled_lr = group['lr']



                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                

                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)



                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)



                p.data.copy_(p_data_fp32)



        return loss
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np 



class GAT_skip_connect(nn.Module):

    def __init__(self, n_node_feat, n_edge_feat, node_embedding_dim, edge_embedding_dim, state_dim, num_layers, concat=True, negative_slope=0.2, dropout=0, bias=True, repeat_edge = 1):

        super(GAT_skip_connect, self).__init__()   

        self.dropout = dropout

        self.negative_slope = negative_slope

        self.n_node_feat = n_node_feat

        self.n_edge_feat = n_edge_feat

        self.state_dim = state_dim

        self.num_layers = num_layers

        self.concat = concat

        self.node_embedding_dim = node_embedding_dim

        self.edge_embedding_dim = edge_embedding_dim

        self.bias = bias

        self.repeat_edge = repeat_edge

        self.Node_embedd = nn.Linear(self.n_node_feat, self.node_embedding_dim, bias = False)

        self.Edge_embedd = nn.Linear(self.n_edge_feat, self.edge_embedding_dim, bias = False)



        self.GATConvs = [GATConvLayer_new(self.node_embedding_dim, self.node_embedding_dim, self.edge_embedding_dim, self.edge_embedding_dim, self.state_dim) for _ in range(num_layers)]



        self.fc1 = nn.Linear(2 * self.node_embedding_dim + self.edge_embedding_dim, 2 * self.node_embedding_dim + self.edge_embedding_dim)

        self.fc2 = nn.Linear(2 * self.node_embedding_dim + self.edge_embedding_dim, self.node_embedding_dim + self.edge_embedding_dim)

        self.fc3 = nn.Linear(self.node_embedding_dim + self.edge_embedding_dim, self.node_embedding_dim + self.edge_embedding_dim)

        self.fc4 = nn.Linear(self.node_embedding_dim + self.edge_embedding_dim, self.edge_embedding_dim)

        self.reg = nn.Linear(self.edge_embedding_dim, 1)



        self.dropout = nn.Dropout(0.1)



    

    def forward(self, x, edge_feat, mask_out):

        len_edges = edge_feat.shape[-1]

        idx_edge = edge_feat[:,len_edges - 1: len_edges]

        mask = torch.where(idx_edge == 0, idx_edge, torch.ones_like(idx_edge).cuda())

        N = x.size(0)

        x = self.Node_embedd(x)

        edge_feat = self.Edge_embedd(edge_feat)

       

        x_ = x

        edge_feat_ = edge_feat



        for i, GATConv in enumerate(self.GATConvs):

            GATConv = GATConv.cuda()

            x, edge_feat = GATConv(x, edge_feat, mask)

            x = F.leaky_relu(x, self.negative_slope)

            edge_feat = F.leaky_relu(edge_feat, self.negative_slope)



            x = x + x_

            edge_feat = edge_feat + edge_feat_



            edge_feat_ = edge_feat

            x_ = x

        

        # concat edge feature and node feature

        out = torch.cat([edge_feat, x.repeat(1, N).view(N * N,  -1), x.repeat(N, 1)], dim = 1)        

        out = F.leaky_relu(self.fc1(out), self.negative_slope) 

        out = F.leaky_relu(self.fc2(out), self.negative_slope)  

        out = F.leaky_relu(self.fc3(out), self.negative_slope)  

        out = F.leaky_relu(self.fc4(out), self.negative_slope)  



        # compute output

        out = self.reg(out)

        out = out * mask

        out = out.reshape(29*29)

        out = out[mask_out]

        return out
from __future__ import division

from __future__ import print_function



import os

import glob

import time

import random

import argparse

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from torch.autograd import Variable



import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import random 

import pandas as pd

import pickle





def log_mae(output, lable):

    output = output.view(-1)

    lable = lable.view(-1)

    loss = torch.abs(output - lable)

    loss = loss.mean()

    loss = torch.log(loss)

    return loss







# Model and optimizer

model = GAT_skip_connect(n_node_feat = 11, n_edge_feat = 26, node_embedding_dim = 256, edge_embedding_dim = 512, state_dim=512, num_layers=3)

# model.load_state_dict(torch.load('../input/maskoutmodel/new_GATConv_3layer_4fcEdgeoutlayer_normout_no_norm_nonzero_maskout_Kagglellpv_v1.pkl'))



model = model.cuda()

model.share_memory()



# optimizer = optim.Adam(model.parameters(), 

#                        lr=0.00001, 

#                        weight_decay=5e-4, amsgrad=False)

optimizer = RAdam(model.parameters(), lr=0.00001, weight_decay=5e-4)

total_loss = {}

best_loss = torch.Tensor([10000000000]).cuda()

best_epoch = 10000





# # Load data, features are tensor 2708, 1433; lables are tensor 2708,; adj is adjacency matrix 2708 * 2708;

in_edges_train = np.load('../input/internalgraphdata/in_edges_train.npz')['arr_0']

nodes_train = np.load('../input/internalgraphdata/nodes_train.npz')['arr_0']

out_edges_train = np.load('../input/extradata/out_edges_train_1.npz', allow_pickle=True)['arr_0']

#train_ = pd.read_csv("../input/champs-scalar-coupling/train.csv")



num_molecules_train = 85003

random.seed(12)

train_total_size = len(nodes_train)

train_size = int(train_total_size * 0.9)

nodes_val = nodes_train[train_size:]

in_edges_val = in_edges_train[train_size:]

out_edges_val = out_edges_train[train_size:]



def train(epoch):

    global best_epoch, best_loss

    model.train()

    loss_epoch = torch.zeros(1).cuda()



    print('--train--')

    for i, node_train in enumerate(nodes_train[:train_size]):

        optimizer.zero_grad()

        

        node_train = torch.Tensor(node_train).cuda()

        node_train = node_train.view(29, -1)

        edge_train = torch.Tensor(in_edges_train[i]).cuda()

        edge_train = edge_train.view(29 * 29, -1)



        out_edge_train = torch.Tensor(out_edges_train[i]).cuda()

        out_edge_train = out_edge_train.view(29 * 29)

        mask_out = out_edge_train.nonzero()

        out_edge_train = out_edge_train[mask_out]

        



        out_edge_train  = Variable(out_edge_train)

        edge_train = Variable(edge_train)

        node_train = Variable(node_train)



        output = model(node_train, edge_train, mask_out)

        loss_train = log_mae(out_edge_train, output)

        loss_train.backward()

        optimizer.step()

        loss_epoch += loss_train.item()

        if i % 1000 == 0:

            print("Train: ")

            print('epoch ', epoch, i, loss_epoch/(i+1))

    loss_epoch /= train_size

    if loss_epoch < best_loss:

        best_loss = loss_epoch

        torch.save(model.state_dict(), 'new_GATConv_3layer_4fcEdgeoutlayer_normout_no_norm_nonzero_maskout_Kagglellpv_v1.pkl')

        best_epoch = epoch



    

    total_loss[epoch] = loss_epoch

    print('Loss epoch ', loss_epoch)



    print('Done ' + str(epoch))







def val(epoch):

    model.eval()

    loss_epoch = torch.zeros(1).cuda()

    print('--validation--')

    with torch.no_grad():

        for i, node_train in enumerate(nodes_val):

            node_train = torch.Tensor(node_train).cuda()

            node_train = node_train.view(29, -1)

            edge_train = torch.Tensor(in_edges_val[i]).cuda()

            edge_train = edge_train.view(29 * 29, -1)

            out_edge_train = torch.Tensor(out_edges_val[i]).cuda()

            out_edge_train = out_edge_train.view(29 * 29)

            mask_out = out_edge_train.nonzero()

            out_edge_train = out_edge_train[mask_out]

            

                

            out_edge_train = Variable(out_edge_train)

            edge_train = Variable(edge_train)

            node_train = Variable(node_train)



            output = model(node_train, edge_train, mask_out)

            loss_train = log_mae(out_edge_train, output)

            loss_epoch += loss_train.item()

            if i % 2000 == 0:

                print('validattion: ')

                print('epoch ', epoch, i, loss_epoch/(i+1))

    loss_epoch /= (train_total_size - train_size)

    print('Loss epoch ', loss_epoch)



for i in range(4):

    train(i)

    val(i)

# train(0)
