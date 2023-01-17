import torch

from torch import nn

from torch.nn import Parameter

import torch.nn.functional as F





 

class GATConvLayer(nn.Module):

    def __init__(self,

                 node_in_fts,

                 node_out_fts,

                 ed_in_fts,

                 ed_out_fts,

                 concat=True,

                 negative_slope=0.1,

                 dropout=0.1,

                 bias=True,

                 repeat_edge = 1

                 ):

        super(GATConvLayer, self).__init__()

 

        self.node_in_fts = node_in_fts

        self.node_out_fts = node_out_fts

        self.ed_out_fts = ed_out_fts

        self.ed_in_fts = ed_in_fts

        self.concat = concat

        self.negative_slope = negative_slope

        self.dropout = dropout

        self.repeat_edge = repeat_edge

 

        self.triplet_transform = Parameter(torch.zeros(size=(2*self.node_in_fts + self.ed_in_fts, self.node_out_fts)))

        self.edge_transform = Parameter(torch.zeros(size=(self.node_out_fts, self.ed_out_fts)))

        self.att = Parameter(torch.Tensor(self.node_out_fts, 1))

        

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

        nn.init.xavier_uniform_(self.triplet_transform.data, gain=1.414)

        nn.init.xavier_uniform_(self.edge_transform.data, gain=1.414)

        nn.init.xavier_uniform_(self.att.data, gain=1.414)

        # glorot(self.att)

        # zeros(self.bias)

 

 

    def forward(self, x, edge_attr):

        return self.propagate( x=x, num_nodes=x.size(0), edge_attr=edge_attr)

 

    def propagate(self, **kwargs):

        edge_attr = kwargs['edge_attr']

        x = kwargs['x']

        N = kwargs['num_nodes']

 

        triplet = torch.cat([x.repeat(1, self.repeat_edge*N).view(self.repeat_edge*N*N, -1), edge_attr, x.repeat(self.repeat_edge*N, 1)], dim=-1)

        new_hij = torch.matmul(triplet, self.triplet_transform) # N^2-fts x fts-out_fts = N^2-out_fts

        new_edge_attr = torch.matmul(new_hij, self.edge_transform) # N^2-fts x edge_fts = N^2-edge_fts

        new_hij = new_hij.view(N, -1, self.node_out_fts) # N-N-fts 

 

        attention = torch.matmul(new_hij, self.att).squeeze() # N-N-fts x fxs-1 = N-N

        attention = F.leaky_relu(attention, self.negative_slope) # N-N

        attention = F.softmax(attention, dim=-1) # N-N

        attention = F.dropout(attention, self.dropout, training=self.training)

 

        h_prime = torch.matmul(attention.view(N, 1, self.repeat_edge*N), new_hij)

        new_x = h_prime.squeeze()

        

        new_x = self.batch_norm_n(new_x)

        new_edge_attr = self.batch_norm_e(new_edge_attr)

 

        return [new_x, new_edge_attr]



class Edge_Update_Only(nn.Module):

    def __init__(self, node_in_fts, edge_in_fts, edge_out_fts, dropout = 0, bias = True, negative_slope = 0.1, concat = True):

        super(self, Edge_Update_Only).__init__()

        self.node_in_fts = node_in_fts

        self.edge_out_fts = edge_out_fts

        self.edge_in_fts = edge_in_fts

        self.dropout = dropout

        self.bias = bias

        self.negative_slope = negative_slope

        self.concat = concat



        self.triplet_transform = nn.Linear(2 * node_in_fts + edge_in_fts, edge_out_fts * 2, bias = False)

        self.FC1 = nn.Linear(self.edge_out_fts * 2, self.edge_out_fts * 2, bias = False)

        self.FC2 = nn.Linear(self.edge_out_fts * 2, self.edge_out_fts, bias = False)



    def forward(node_fts, edge_fts):

        N = node_fts.size(1)

        triplet = torch.cat([node_fts.repeat(1, 1, N).view(-1, N * N, self.node_fts), x.repeat(1, N, 1), edge_fts], dim = -1)

        new_edge = self.triplet_transform(triplet)
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np 



class GAT(nn.Module):

    def __init__(self, n_node_feat, n_edge_feat, node_embedding_dim, edge_embedding_dim, concat=True, negative_slope=0.2, dropout=0, bias=True, repeat_edge = 1):

        super(GAT, self).__init__()   

        self.dropout = dropout

        self.negative_slope = negative_slope

        self.n_node_feat = n_node_feat

        self.n_edge_feat = n_edge_feat

        self.concat = concat

        self.node_embedding_dim = node_embedding_dim

        self.edge_embedding_dim = edge_embedding_dim

        self.bias = bias

        self.repeat_edge = repeat_edge

        self.Node_embedd = nn.Linear(self.n_node_feat, self.node_embedding_dim, bias = False)

        self.Edge_embedd = nn.Linear(self.n_edge_feat, self.edge_embedding_dim, bias = False)



        self.GATConv1 = GATConvLayer(self.node_embedding_dim, self.node_embedding_dim, self.edge_embedding_dim, self.edge_embedding_dim)

        self.GATConv2 = GATConvLayer(self.node_embedding_dim, self.node_embedding_dim, self.edge_embedding_dim, self.edge_embedding_dim)

        self.GATConv3 = GATConvLayer(self.node_embedding_dim, self.node_embedding_dim, self.edge_embedding_dim, self.edge_embedding_dim)



        

        self.fc1 = nn.Linear(2 * self.node_embedding_dim + self.edge_embedding_dim, 2 * self.node_embedding_dim + self.edge_embedding_dim)

        self.fc2 = nn.Linear(2 * self.node_embedding_dim + self.edge_embedding_dim, 2 * self.node_embedding_dim + self.edge_embedding_dim)

        self.reg = nn.Linear(2 * self.node_embedding_dim + self.edge_embedding_dim, 1)



        self.dropout = nn.Dropout(0.1)



    

    def forward(self, x, edge_feat):

        len_edges = edge_feat.shape[-1]

        idx_edge = edge_feat[:,len_edges - 1: len_edges]

        mask = torch.where(idx_edge == 0, idx_edge, torch.ones_like(idx_edge).cuda())

        N = x.size(0)

        x = self.Node_embedd(x)

        edge_feat = self.Edge_embedd(edge_feat)

        x, edge_feat = self.GATConv1(x, edge_feat)

        x = F.leaky_relu(x, self.negative_slope)

        edge_feat = F.leaky_relu(edge_feat, self.negative_slope)



        x, edge_feat = self.GATConv2(x, edge_feat)

        x = F.leaky_relu(x, self.negative_slope)

        edge_feat = F.leaky_relu(edge_feat, self.negative_slope)



        x, edge_feat = self.GATConv3(x, edge_feat)

        x = F.leaky_relu(x, self.negative_slope)

        edge_feat = F.leaky_relu(edge_feat, self.negative_slope)

                 

        # transform node feature N x node_feature to N^2 x node_features matrix

        trans = x.repeat(1, N)

        trans = trans.view(N * N,  -1)

        trans = torch.cat([trans, x.repeat(N, 1)], dim = 1)

       

        trans = trans.cuda()

        # concat edge feature and node feature

        out = torch.cat([edge_feat, trans], dim = 1)        

        out = F.leaky_relu(self.fc1(out), self.negative_slope) 

        out = self.dropout(out)

        out = F.leaky_relu(self.fc2(out), self.negative_slope)  

        out = self.dropout(out)



        # compute output

        out = self.reg(out)

        out = out * mask

        return out

    # def reset_parameters(self):

    #     # nn.init.xavier_uniform_(self.a, gain=1.414)

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

model = GAT(n_node_feat = 11, n_edge_feat = 26, node_embedding_dim = 256, edge_embedding_dim = 512)

model.load_state_dict(torch.load('../input/newmodel/new_model_3conv_256_log_mae_no_norm_RAdam_Kaggle_v2.pkl'))



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

train_size = int(train_total_size * 0.95)

nodes_val = nodes_train[train_size:]

in_edges_val = in_edges_train[train_size:]

out_edges_val = out_edges_train[train_size:]



def train(epoch):

    global best_epoch, best_loss

    model.train()

    loss_epoch = torch.zeros(1).cuda()



    print('--train--')

    for i, node_train in enumerate(nodes_train):

        optimizer.zero_grad()

        

        node_train = torch.Tensor(node_train).cuda()

        node_train = node_train.view(29, -1)

        edge_train = torch.Tensor(in_edges_train[i]).cuda()

        edge_train = edge_train.view(29 * 29, -1)



        out_edge_train = torch.Tensor(out_edges_train[i]).cuda()

        out_edge_train = out_edge_train.view(29 * 29, -1)



        out_edge_train  = Variable(out_edge_train)

        edge_train = Variable(edge_train)

        node_train = Variable(node_train)



        output = model(node_train, edge_train)

        loss_train = log_mae(out_edge_train, output)

        loss_train.backward()

        optimizer.step()

        

        loss_epoch += loss_train.item()

        if i % 1000 == 0:

            print("Train: ")

            print('epoch ', epoch, i, loss_epoch/(i+1))

    loss_epoch /= 85003

    if loss_epoch < best_loss:

        best_loss = loss_epoch

        torch.save(model.state_dict(), 'new_model_3conv_256_log_mae_no_norm_RAdam_Kaggle_v3.pkl')

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

            out_edge_train = out_edge_train.view(29 * 29, -1)

            

            out_edge_train = Variable(out_edge_train)

            edge_train = Variable(edge_train)

            node_train = Variable(node_train)



            output = model(node_train, edge_train)

            loss_train = F.l1_loss(out_edge_train, output)

            loss_epoch += loss_train.item()

            if i % 1000 == 0:

                print('validattion: ')

                print('epoch ', epoch, i, loss_epoch/(i+1))

    loss_epoch /= (train_total_size - train_size)

    print('Loss epoch ', loss_epoch)



for i in range(25):

    train(i)

    val(i)


