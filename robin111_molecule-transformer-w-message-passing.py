import copy

import math

import random

import pandas as pd

import numpy as np

from functools import partial



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from fastai.callbacks import SaveModelCallback

from fastai.basic_data import DataBunch, DeviceDataLoader, DatasetType

from fastai.basic_train import Learner, LearnerCallback, Callback, add_metrics

from fastai.train import *
# Constants

TYPES           = np.array(['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', 

                            '3JHC', '3JHN'])

TYPES_MAP      = {t: i for i, t in enumerate(TYPES)}





SC_EDGE_FEATS = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 

                 'type_6', 'type_7', 'dist', 'dist_min_rad', 

                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 

                 'cos_angle0', 'cos_angle1']

SC_MOL_FEATS  = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 

                 'type_6', 'type_7', 'dist', 'dist_min_rad', 

                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 

                 'cos_angle0', 'cos_angle1', 'num_atoms', 'num_C_atoms', 

                 'num_F_atoms', 'num_H_atoms', 'num_N_atoms', 'num_O_atoms', 

                 'std_bond_length', 'ave_bond_length', 'ave_atom_weight']

ATOM_FEATS    = ['type_H', 'type_C', 'type_N', 'type_O', 'type_F', 'degree_1', 

                 'degree_2', 'degree_3', 'degree_4', 'degree_5', 'SP', 'SP2', 

                 'SP3', 'hybridization_unspecified', 'aromatic', 

                 'formal_charge', 'atomic_num', 'ave_bond_length', 

                 'ave_neighbor_weight', 'donor', 'acceptor']

BOND_FEATS    = ['single', 'double', 'triple', 'aromatic', 'conjugated', 

                 'in_ring', 'dist', 'normed_dist']





TARGET_COL   = 'scalar_coupling_constant'

CONTRIB_COLS = ['fc', 'sd', 'pso', 'dso']





N_TYPES            = 8

N_SC_EDGE_FEATURES = 16

N_SC_MOL_FEATURES  = 25

N_ATOM_FEATURES    = 21

N_BOND_FEATURES    = 8

MAX_N_ATOMS        = 29

MAX_N_SC           = 135

BATCH_PAD_VAL      = -999

N_MOLS             = 130775





N_FOLDS = 8





SC_MEAN             = 16

SC_STD              = 35

SC_FEATS_TO_SCALE   = ['dist', 'dist_min_rad', 'dist_electro_neg_adj', 

                       'num_atoms', 'num_C_atoms', 'num_F_atoms', 'num_H_atoms', 

                       'num_N_atoms', 'num_O_atoms', 'ave_bond_length', 

                       'std_bond_length', 'ave_atom_weight']

ATOM_FEATS_TO_SCALE = ['atomic_num', 'ave_bond_length', 'ave_neighbor_weight']

BOND_FEATS_TO_SCALE = ['dist']





RAW_DATA_PATH  = '../input/champs-scalar-coupling/'

PROC_DATA_PATH = '../input/champs-proc-data/'
# Define some helper functions and general classes

def set_seed(seed=100):

    """Set the seed for all relevant RNGs."""

    # python RNG

    random.seed(seed)



    # pytorch RNGs

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



    # numpy RNG

    np.random.seed(seed)



def scale_features(df, features, train_mol_ids=None, means=None, stds=None,

                   return_mean_and_std=False):

    if ((df[features].mean().abs()>0.1).any()

        or ((df[features].std()-1.0).abs()>0.1).any()):

        if train_mol_ids is not None:

            idx = df['molecule_id'].isin(train_mol_ids)

            means = df.loc[idx, features].mean()

            stds = df.loc[idx, features].std()

        else:

            assert means is not None

            assert stds is not None

        df[features] = (df[features] - means) / stds

    if return_mean_and_std: return df, means, stds

    else: return df



def scatter_add(src, idx, num=None, dim=0, out=None):

    """Adds all elements from 'src' into 'out' at the positions specified by 

    'idx'. The index 'idx' only has to match the size of 'src' in dimension 

    'dim'. If 'out' is None it is initialized to zeros of size 'num' along 'dim' 

    and of equal dimension to 'src' at all other dimensions."""

    if not num: num = idx.max().item() + 1

    sz, expanded_idx_sz = src.size(), src.size()

    sz = sz[:dim] + torch.Size((num,)) + sz[(dim+1):]

    expanded_idx = idx.unsqueeze(-1).expand(expanded_idx_sz)

    if out is None: out = torch.zeros(sz, dtype=src.dtype, device=src.device)

    return out.scatter_add(dim, expanded_idx, src)



def scatter_mean(src, idx, num=None, dim=0, out=None):

    return (scatter_add(src, idx, num, dim, out) 

            / scatter_add(torch.ones_like(src), idx, num, dim).clamp(1.0))



# The below layernorm class initializes parameters according to the default 

# initialization of bacthnorm layers in pytorch v1.1 and below. Somehow this 

# initialization seemed to work beter.

class LayerNorm(nn.LayerNorm):

    """Class overriding pytorch default layernorm intitialization."""

    def reset_parameters(self):

        if self.elementwise_affine:

            nn.init.uniform_(self.weight)

            nn.init.zeros_(self.bias)



def hidden_layer(d_in, d_out, batch_norm, dropout, layer_norm=False, act=None):

    layers = []

    layers.append(nn.Linear(d_in, d_out))

    if act: layers.append(act)

    if batch_norm: layers.append(nn.BatchNorm1d(d_out))

    if layer_norm: layers.append(LayerNorm(d_out))

    if dropout != 0: layers.append(nn.Dropout(dropout))

    return layers



class FullyConnectedNet(nn.Module):

    """General purpose neural network class with fully connected layers."""

    def __init__(self, d_input, d_output=None, layers=[], act=nn.ReLU(True), 

                 dropout=[], batch_norm=False, out_act=None, final_bn=False, 

                 layer_norm=False, final_ln=False):

        super().__init__()

        sizes = [d_input] + layers

        if d_output: 

            sizes += [d_output]

            dropout += [0.0]

        layers_ = []

        for i, (d_in, d_out, dr) in enumerate(zip(sizes[:-1], sizes[1:], dropout)):

            act_ = act if i < len(layers) else out_act

            batch_norm_ = batch_norm if i < len(layers) else final_bn

            layer_norm_ = layer_norm if i < len(layers) else final_ln

            layers_ += hidden_layer(d_in, d_out, batch_norm_, dr, layer_norm_, act_)      

        self.layers = nn.Sequential(*layers_)

        

    def forward(self, x):

        return self.layers(x)
fold_id, version = 1, 1

model_str = f'mol_transformer_v{version}_fold{fold_id}'
train_df = pd.read_csv(PROC_DATA_PATH+'train_proc_df.csv', index_col=0)

test_df  = pd.read_csv(PROC_DATA_PATH+'test_proc_df.csv', index_col=0)

atom_df  = pd.read_csv(PROC_DATA_PATH+'atom_df.csv', index_col=0)

bond_df  = pd.read_csv(PROC_DATA_PATH+'bond_df.csv', index_col=0)

angle_in_df   = pd.read_csv(PROC_DATA_PATH+'angle_in_df.csv', index_col=0)

angle_out_df  = pd.read_csv(PROC_DATA_PATH+'angle_out_df.csv', index_col=0)

graph_dist_df = pd.read_csv(PROC_DATA_PATH+'graph_dist_df.csv', index_col=0, dtype=np.int32)

structures_df = pd.read_csv(PROC_DATA_PATH+'structures_proc_df.csv', index_col=0)



train_mol_ids = pd.read_csv(PROC_DATA_PATH+'train_idxs_8_fold_cv.csv',

                            usecols=[0, fold_id], index_col=0

                            ).dropna().astype(int).iloc[:,0]

val_mol_ids   = pd.read_csv(PROC_DATA_PATH+'val_idxs_8_fold_cv.csv',

                            usecols=[0, fold_id], index_col=0

                            ).dropna().astype(int).iloc[:,0]

test_mol_ids  = pd.Series(test_df['molecule_id'].unique())





# scale features

train_df, sc_feat_means, sc_feat_stds = scale_features(

    train_df, SC_FEATS_TO_SCALE, train_mol_ids, return_mean_and_std=True)

test_df = scale_features(

    test_df, SC_FEATS_TO_SCALE, means=sc_feat_means, stds=sc_feat_stds)

atom_df = scale_features(atom_df, ATOM_FEATS_TO_SCALE, train_mol_ids)

bond_df = scale_features(bond_df, BOND_FEATS_TO_SCALE, train_mol_ids)





# group data by molecule id

gb_mol_sc = train_df.groupby('molecule_id')

test_gb_mol_sc = test_df.groupby('molecule_id')

gb_mol_atom = atom_df.groupby('molecule_id')

gb_mol_bond = bond_df.groupby('molecule_id')

gb_mol_struct = structures_df.groupby('molecule_id')

gb_mol_angle_in = angle_in_df.groupby('molecule_id')

gb_mol_angle_out = angle_out_df.groupby('molecule_id')

gb_mol_graph_dist = graph_dist_df.groupby('molecule_id')
def clones(module, N):

    """Produce N identical layers."""

    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class SublayerConnection(nn.Module):

    """

    A residual connection followed by a layer norm.

    Note for code simplicity the norm is first as opposed to last.

    """

    def __init__(self, size, dropout):

        super().__init__()

        self.norm = LayerNorm(size)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x, sublayer):

        """Apply residual connection to any sublayer with the same size."""

        return x + self.dropout(sublayer(self.norm(x)))

    

def _gather_nodes(x, idx, sz_last_dim):

    idx = idx.unsqueeze(-1).expand(-1, -1, sz_last_dim)

    return x.gather(1, idx)



class ENNMessage(nn.Module):

    """

    The edge network message passing function from the MPNN paper. Optionally 

    adds and additional cosine angle based attention mechanism over incoming

    messages.

    """

    PAD_VAL = -999

    

    def __init__(self, d_model, d_edge, kernel_sz, enn_args={}, ann_args=None):

        super().__init__()

        assert kernel_sz <= d_model

        self.d_model, self.kernel_sz = d_model, kernel_sz

        self.enn = FullyConnectedNet(d_edge, d_model*kernel_sz, **enn_args)

        if ann_args: self.ann = FullyConnectedNet(1, d_model, **ann_args)

        else: self.ann = None

    

    def forward(self, x, edges, pairs_idx, angles=None, angles_idx=None, t=0):

        """Note that edges and pairs_idx raw inputs are for a unidirectional 

        graph. They are expanded to allow bidirectional message passing.""" 

        if t==0: 

            self.set_a_mat(edges)

            if self.ann: self.set_attn(angles)

            # concat reversed pairs_idx for bidirectional message passing

            self.pairs_idx = torch.cat([pairs_idx, pairs_idx[:,:,[1,0]]], dim=1)

        return self.add_message(torch.zeros_like(x), x, angles_idx)

    

    def set_a_mat(self, edges):

        n_edges = edges.size(1)

        a_vect = self.enn(edges) 

        a_vect = a_vect / (self.kernel_sz ** .5) # rescale

        mask = edges[:,:,0,None].expand(a_vect.size())==self.PAD_VAL

        a_vect = a_vect.masked_fill(mask, 0.0)

        self.a_mat = a_vect.view(-1, n_edges, self.d_model, self.kernel_sz)

        # concat a_mats for bidirectional message passing

        self.a_mat = torch.cat([self.a_mat, self.a_mat], dim=1)

    

    def set_attn(self, angles):

        angles = angles.unsqueeze(-1)

        self.attn = self.ann(angles)

        mask = angles.expand(self.attn.size())==self.PAD_VAL

        self.attn = self.attn.masked_fill(mask, 0.0)

    

    def add_message(self, m, x, angles_idx=None):

        """Add message for atom_{i}: m_{i} += sum_{j}[attn_{ij} A_{ij}x_{j}]."""

        # select the 'x_{j}' feeding into the 'm_{i}'

        x_in = _gather_nodes(x, self.pairs_idx[:,:,1], self.d_model)

        

        # do the matrix multiplication 'A_{ij}x_{j}'

        if self.kernel_sz==self.d_model: # full matrix multiplcation

            ax = (x_in.unsqueeze(-2) @ self.a_mat).squeeze(-2)

        else: # do a convolution

            x_padded = F.pad(x_in, self.n_pad)

            x_unfolded = x_padded.unfold(-1, self.kernel_sz, 1)

            ax = (x_unfolded * self.a_mat).sum(-1)

        

        # apply atttention

        if self.ann:

            n_pairs = self.pairs_idx.size(1)

            # average all attn(angle_{ijk}) per edge_{ij}. 

            # i.e.: attn_{ij} = sum_{k}[attn(angle_{ijk})] / n_angles_{ij}

            ave_att = scatter_mean(self.attn, angles_idx, num=n_pairs, dim=1, 

                                   out=torch.ones_like(ax))

            ax = ave_att * ax

        

        # sum up all 'A_{ij}h_{j}' per node 'i'

        idx_0 = self.pairs_idx[:,:,0,None].expand(-1, -1, self.d_model)

        return m.scatter_add(1, idx_0, ax)

    

    @property

    def n_pad(self):

        k = self.kernel_sz

        return (k // 2, k // 2 - int(k % 2 == 0))



class MultiHeadedDistAttention(nn.Module):

    """Generalizes the euclidean and graph distance based attention layers."""

    def __init__(self, h, d_model):

        super().__init__()

        self.d_model, self.d_k, self.h = d_model, d_model // h, h

        self.attn = None

        self.linears = clones(nn.Linear(d_model, d_model), 2)

        

    def forward(self, dists, x, mask):

        batch_size = x.size(0)

        x = self.linears[0](x).view(batch_size, -1, self.h, self.d_k)

        x, self.attn = self.apply_attn(dists, x, mask)

        x = x.view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](x)

    

    def apply_attn(self, dists, x, mask):

        attn = self.create_raw_attn(dists, mask)

        attn = attn.transpose(-2,-1).transpose(1, 2)

        x = x.transpose(1, 2)

        x = torch.matmul(attn, x)

        x = x.transpose(1, 2).contiguous()

        return x, attn

    

    def create_raw_attn(self, dists, mask):

        pass



class MultiHeadedGraphDistAttention(MultiHeadedDistAttention):

    """Attention based on an embedding of the graph distance matrix."""

    MAX_GRAPH_DIST = 10

    def __init__(self, h, d_model):

        super().__init__(h, d_model)

        self.embedding = nn.Embedding(self.MAX_GRAPH_DIST+1, h)

    

    def create_raw_attn(self, dists, mask):

        emb_dists = self.embedding(dists)

        mask = mask.unsqueeze(-1).expand(emb_dists.size())

        emb_dists = emb_dists.masked_fill(mask==0, -1e9)

        return F.softmax(emb_dists, dim=-2).masked_fill(mask==0, 0)



class MultiHeadedEuclDistAttention(MultiHeadedDistAttention):

    """Attention based on a parameterized normal pdf taking a molecule's 

    euclidean distance matrix as input."""

    def __init__(self, h, d_model):

        super().__init__(h, d_model)

        self.log_prec = nn.Parameter(torch.Tensor(1, 1, 1, h))

        self.locs = nn.Parameter(torch.Tensor(1, 1, 1, h))

        nn.init.normal_(self.log_prec, mean=0.0, std=0.1)

        nn.init.normal_(self.locs, mean=0.0, std=1.0)

    

    def create_raw_attn(self, dists, mask):

        dists = dists.unsqueeze(-1).expand(-1, -1, -1, self.h)

        z = torch.exp(self.log_prec) * (dists - self.locs)

        pdf = torch.exp(-0.5 * z ** 2)

        return pdf / pdf.sum(dim=-2, keepdim=True).clamp(1e-9)      



def attention(query, key, value, mask=None, dropout=None):

    """Compute 'Scaled Dot Product Attention'."""

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None: scores = scores.masked_fill(mask==0, -1e9)

    p_attn = F.softmax(scores, dim=-1).masked_fill(mask==0, 0)

    if dropout is not None: p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn



class MultiHeadedSelfAttention(nn.Module):

    """Applies self-attention as described in the Transformer paper."""

    def __init__(self, h, d_model, dropout=0.1):

        super().__init__()

        self.d_model, self.d_k, self.h = d_model, d_model // h, h

        self.attn = None

        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        

    def forward(self, x, mask):

        # Same mask applied to all h heads.

        mask = mask.unsqueeze(1)

        batch_size = x.size(0)

        

        # 1) Do all the linear projections in batch from d_model => h x d_k 

        query, key, value = [

            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

            for l in self.linears[:3]

        ]

        

        # 2) Apply attention on all the projected vectors in batch. 

        x, self.attn = attention(query, key, value, mask, self.dropout)

        

        # 3) "Concat" using a view and apply a final linear. 

        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, self.d_model)

        return self.linears[-1](x)



class AttendingLayer(nn.Module):

    """Stacks the three attention layers and the pointwise feedforward net."""

    def __init__(self, size, eucl_dist_attn, graph_dist_attn, self_attn, ff, dropout):

        super().__init__()

        self.eucl_dist_attn = eucl_dist_attn

        self.graph_dist_attn = graph_dist_attn

        self.self_attn = self_attn

        self.ff = ff

        self.subconns = clones(SublayerConnection(size, dropout), 4)

        self.size = size



    def forward(self, x, eucl_dists, graph_dists, mask):

        eucl_dist_sub = lambda x: self.eucl_dist_attn(eucl_dists, x, mask)

        x = self.subconns[0](x, eucl_dist_sub)

        graph_dist_sub = lambda x: self.graph_dist_attn(graph_dists, x, mask)

        x = self.subconns[1](x, graph_dist_sub)

        self_sub = lambda x: self.self_attn(x, mask)

        x = self.subconns[2](x, self_sub)

        return self.subconns[3](x, self.ff)



class MessagePassingLayer(nn.Module):

    """Stacks the bond and scalar coupling pair message passing layers."""

    def __init__(self, size, bond_mess, sc_mess, dropout, N):

        super().__init__()

        self.bond_mess = bond_mess

        self.sc_mess = sc_mess

        self.linears = clones(nn.Linear(size, size), 2*N)

        self.subconns = clones(SublayerConnection(size, dropout), 2*N)



    def forward(self, x, bond_x, sc_pair_x, angles, mask, bond_idx, sc_idx, 

                angles_idx, t=0):

        bond_sub = lambda x: self.linears[2*t](

            self.bond_mess(x, bond_x, bond_idx, angles, angles_idx, t))

        x = self.subconns[2*t](x, bond_sub)

        sc_sub = lambda x: self.linears[(2*t)+1](self.sc_mess(x, sc_pair_x, sc_idx, t=t))

        return self.subconns[(2*t)+1](x, sc_sub)



class Encoder(nn.Module):

    """Encoder stacks N attention layers and one message passing layer."""

    def __init__(self, mess_pass_layer, attn_layer, N):

        super().__init__()

        self.mess_pass_layer = mess_pass_layer

        self.attn_layers = clones(attn_layer, N)

        self.norm = LayerNorm(attn_layer.size)

        

    def forward(self, x, bond_x, sc_pair_x, eucl_dists, graph_dists, angles, 

                mask, bond_idx, sc_idx, angles_idx):

        """Pass the inputs (and mask) through each block in turn. Note that for 

        each block the same message passing layer is used."""

        for t, attn_layer in enumerate(self.attn_layers):

            x = self.mess_pass_layer(x, bond_x, sc_pair_x, angles, mask, 

                                     bond_idx, sc_idx, angles_idx, t)

            x = attn_layer(x, eucl_dists, graph_dists, mask)

        return self.norm(x)
def create_contrib_head(d_in, d_ff, act, dropout=0.0, layer_norm=True):

    layers = hidden_layer(d_in, d_ff, False, dropout, layer_norm, act)

    layers += hidden_layer(d_ff, 1, False, 0.0) # output layer

    return nn.Sequential(*layers)



class ContribsNet(nn.Module):

    """The feed-forward net used for the sc contribution and final sc constant predictions."""

    N_CONTRIBS = 5

    CONTIB_SCALES = [1, 250, 45, 35, 500] # scales used to make the 5 predictions of similar magnitude

    

    def __init__(self, d_in, d_ff, vec_in, act, dropout=0.0, layer_norm=True):

        super().__init__()

        contrib_head = create_contrib_head(d_in, d_ff, act, dropout, layer_norm) 

        self.blocks = clones(contrib_head, self.N_CONTRIBS)

        

    def forward(self, x):

        ys = torch.cat(

            [b(x)/s for b,s in zip(self.blocks, self.CONTIB_SCALES)], dim=-1)

        return torch.cat([ys[:,:-1], ys.sum(dim=-1, keepdim=True)], dim=-1)

    

class MyCustomHead(nn.Module):

    """Joins the sc type specific residual block with the sc contribution 

    feed-forward net."""

    PAD_VAL = -999

    N_TYPES = 8

    

    def __init__(self, d_input, d_ff, d_ff_contribs, pre_layers=[], post_layers=[], 

                 act=nn.ReLU(True), dropout=3*[0.], norm=False):

        super().__init__()

        fc_pre = hidden_layer(d_input, d_ff, False, dropout[0], norm, act)

        self.preproc = nn.Sequential(*fc_pre)

        fc_type = hidden_layer(d_ff, d_input, False, dropout[1], norm, act)

        self.types_net = clones(nn.Sequential(*fc_type), self.N_TYPES)

        self.contribs_net = ContribsNet(

            d_input, d_ff_contribs, d_ff, act, dropout[2], layer_norm=norm)

        

    def forward(self, x, sc_types):

        # stack inputs with a .view for easier processing

        x, sc_types = x.view(-1, x.size(-1)), sc_types.view(-1)

        mask =  sc_types != self.PAD_VAL

        x, sc_types = x[mask], sc_types[mask]

        

        x_ = self.preproc(x)

        x_types = torch.zeros_like(x)

        for i in range(self.N_TYPES):

            t_idx = sc_types==i

            if torch.any(t_idx): x_types[t_idx] = self.types_net[i](x_[t_idx])

        x = x + x_types 

        return self.contribs_net(x)
class Transformer(nn.Module):

    """Molecule transformer with message passing."""

    def __init__(self, d_atom, d_bond, d_sc_pair, d_sc_mol, N=6, d_model=512, 

                 d_ff=2048, d_ff_contrib=128, h=8, dropout=0.1, kernel_sz=128, 

                 enn_args={}, ann_args={}):

        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model

        c = copy.deepcopy

        bond_mess = ENNMessage(d_model, d_bond, kernel_sz, enn_args, ann_args)

        sc_mess = ENNMessage(d_model, d_sc_pair, kernel_sz, enn_args)

        eucl_dist_attn = MultiHeadedEuclDistAttention(h, d_model)

        graph_dist_attn = MultiHeadedGraphDistAttention(h, d_model)

        self_attn = MultiHeadedSelfAttention(h, d_model, dropout)

        ff = FullyConnectedNet(d_model, d_model, [d_ff], dropout=[dropout])

        

        message_passing_layer = MessagePassingLayer(d_model, bond_mess, sc_mess, dropout, N)

        attending_layer = AttendingLayer(d_model, c(eucl_dist_attn), c(graph_dist_attn), 

                                         c(self_attn), c(ff), dropout)

        

        self.projection = nn.Linear(d_atom, d_model)

        self.encoder = Encoder(message_passing_layer, attending_layer, N)

        self.write_head = MyCustomHead(2*d_model+d_sc_mol, d_ff, d_ff_contrib, norm=True)

        

    def forward(self, atom_x, bond_x, sc_pair_x, sc_mol_x, eucl_dists, graph_dists, 

                angles, mask, bond_idx, sc_idx, angles_idx, sc_types):

        x = self.encoder(

            self.projection(atom_x), bond_x, sc_pair_x, eucl_dists, graph_dists, 

            angles, mask, bond_idx, sc_idx, angles_idx

        )

        # for each sc constant in the batch select and concat the relevant pairs 

        # of atom  states.

        x = torch.cat(

            [_gather_nodes(x, sc_idx[:,:,0], self.d_model), 

             _gather_nodes(x, sc_idx[:,:,1], self.d_model), 

             sc_mol_x], dim=-1

        )

        return self.write_head(x, sc_types)
def _get_existing_group(gb, i):

    try: group_df = gb.get_group(i)

    except KeyError: group_df = None

    return group_df



def get_dist_matrix(struct_df):

    locs = struct_df[['x','y','z']].values

    n_atoms = len(locs)

    loc_tile = np.tile(locs.T, (n_atoms,1,1))

    dist_mat = np.sqrt(((loc_tile - loc_tile.T)**2).sum(axis=1))

    return dist_mat



class MoleculeDataset(Dataset):

    """Dataset returning inputs and targets per molecule."""

    def __init__(self, mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, 

                 gb_mol_struct, gb_mol_angle_in, gb_mol_angle_out, 

                 gb_mol_graph_dist):

        """Dataset is constructed from dataframes grouped by molecule_id."""

        self.n = len(mol_ids)

        self.mol_ids = mol_ids

        self.gb_mol_sc = gb_mol_sc

        self.gb_mol_atom = gb_mol_atom

        self.gb_mol_bond = gb_mol_bond

        self.gb_mol_struct = gb_mol_struct

        self.gb_mol_angle_in = gb_mol_angle_in

        self.gb_mol_angle_out = gb_mol_angle_out

        self.gb_mol_graph_dist = gb_mol_graph_dist



    def __len__(self):

        return self.n



    def __getitem__(self, idx):

        return (self.gb_mol_sc.get_group(self.mol_ids[idx]),

                self.gb_mol_atom.get_group(self.mol_ids[idx]), 

                self.gb_mol_bond.get_group(self.mol_ids[idx]), 

                self.gb_mol_struct.get_group(self.mol_ids[idx]), 

                self.gb_mol_angle_in.get_group(self.mol_ids[idx]), 

                _get_existing_group(self.gb_mol_angle_out, self.mol_ids[idx]),

                self.gb_mol_graph_dist.get_group(self.mol_ids[idx]))



def arr_lst_to_padded_batch(arr_lst, dtype=torch.float, pad_val=BATCH_PAD_VAL):

    tensor_list = [torch.Tensor(arr).type(dtype) for arr in arr_lst]

    batch = torch.nn.utils.rnn.pad_sequence(

        tensor_list, batch_first=True, padding_value=pad_val)

    return batch.contiguous()

                   

def collate_parallel_fn(batch, test=False):

    """

    Transforms input dataframes grouped by molecule into a batch of input and 

    target tensors for a 'batch_size' number of molecules. The first dimension 

    is used as the batch dimension.



    Returns:

        - atom_x: features at the atom level

        - bond_x: features at the chemical bond level

        - sc_x: features describing the scalar coupling atom_0 and atom_1 pairs

        - sc_m_x: in addition to the set of features in 'sc_x', includes 

            features at the molecule level.

        - eucl_dists: 3D euclidean distance matrices

        - graph_dists: graph distance matrices

        - angles: cosine angles between all chemical bonds

        - mask: binary mask of dim=(batch_size, max_n_atoms, max_n_atoms),

            where max_n_atoms is the largest number of atoms per molecule in 

            'batch'

        - bond_idx: tensor of dim=(batch_size, max_n_bonds, 2), containing the

            indices of atom_0 and atom_1 pairs that form chemical bonds

        - sc_idx: tensor of dim=(batch_size, max_n_sc, 2), containing the

            indices of atom_0 and atom_1 pairs that form a scalar coupling

            pair

        - angles_idx: tensor of dim=(batch_size, max_n_angles, 1), mapping 

            angles to the chemical bonds in the molecule.

        - sc_types: scalar coupling types

        - sc_vals: scalar coupling contributions (first 4 columns) and constant

            (last column)

    """

    batch_size, n_atom_sum, n_pairs_sum = len(batch), 0, 0

    atom_x, bond_x, sc_x, sc_m_x = [], [], [], []

    eucl_dists, graph_dists = [], []

    angles_in, angles_out = [], []

    mask, bond_idx, sc_idx = [], [], []

    angles_in_idx, angles_out_idx = [], []

    sc_types, sc_vals = [], []



    for b in range(batch_size):

        (sc_df, atom_df, bond_df, struct_df, angle_in_df, angle_out_df, 

         graph_dist_df) = batch[b]

        n_atoms, n_pairs, n_sc = len(atom_df), len(bond_df), len(sc_df)

        n_pad = MAX_N_ATOMS - n_atoms

        eucl_dists_ = get_dist_matrix(struct_df)

        eucl_dists_ = np.pad(eucl_dists_, [(0, 0), (0, n_pad)], 'constant', 

                             constant_values=999)

        

        atom_x.append(atom_df[ATOM_FEATS].values)

        bond_x.append(bond_df[BOND_FEATS].values)

        sc_x.append(sc_df[SC_EDGE_FEATS].values)

        sc_m_x.append(sc_df[SC_MOL_FEATS].values)

        sc_types.append(sc_df['type'].values)

        if not test: 

            n_sc_pad = MAX_N_SC - n_sc

            sc_vals_ = sc_df[CONTRIB_COLS+[TARGET_COL]].values

            sc_vals.append(np.pad(sc_vals_, [(0, n_sc_pad), (0, 0)], 'constant', 

                                  constant_values=-999))

        eucl_dists.append(eucl_dists_)

        graph_dists.append(graph_dist_df.values[:,:-1])

        angles_in.append(angle_in_df['cos_angle'].values)

        if angle_out_df is not None: 

            angles_out.append(angle_out_df['cos_angle'].values)

        else: 

            angles_out.append(np.array([BATCH_PAD_VAL]))

        

        mask.append(np.pad(np.ones(2 * [n_atoms]), [(0, 0), (0, n_pad)], 

                           'constant'))

        bond_idx.append(bond_df[['idx_0', 'idx_1']].values)

        sc_idx.append(sc_df[['atom_index_0', 'atom_index_1']].values)

        angles_in_idx.append(angle_in_df['b_idx'].values)

        if angle_out_df is not None: 

            angles_out_idx.append(angle_out_df['b_idx'].values)

        else:

            angles_out_idx.append(np.array([0.]))

        

        n_atom_sum += n_atoms

        n_pairs_sum += n_pairs

        

    atom_x = arr_lst_to_padded_batch(atom_x, pad_val=0.)

    bond_x = arr_lst_to_padded_batch(bond_x)

    max_n_atoms = atom_x.size(1)

    max_n_bonds = bond_x.size(1)

    angles_out_idx = [a + max_n_bonds for a in angles_out_idx]

    

    sc_x = arr_lst_to_padded_batch(sc_x)

    sc_m_x =arr_lst_to_padded_batch(sc_m_x)

    if not test: sc_vals = arr_lst_to_padded_batch(sc_vals)

    else: sc_vals = torch.tensor([0.] * batch_size)

    sc_types = arr_lst_to_padded_batch(sc_types, torch.long)

    mask = arr_lst_to_padded_batch(mask, torch.uint8, 0)

    mask = mask[:,:,:max_n_atoms].contiguous()

    bond_idx = arr_lst_to_padded_batch(bond_idx, torch.long, 0)

    sc_idx = arr_lst_to_padded_batch(sc_idx, torch.long, 0)

    angles_in_idx = arr_lst_to_padded_batch(angles_in_idx, torch.long, 0)

    angles_out_idx = arr_lst_to_padded_batch(angles_out_idx, torch.long, 0)

    angles_idx = torch.cat((angles_in_idx, angles_out_idx), dim=-1).contiguous()

    eucl_dists = arr_lst_to_padded_batch(eucl_dists, pad_val=999)

    eucl_dists = eucl_dists[:,:,:max_n_atoms].contiguous()

    graph_dists = arr_lst_to_padded_batch(graph_dists, torch.long, 10)

    graph_dists = graph_dists[:,:,:max_n_atoms].contiguous()

    angles_in = arr_lst_to_padded_batch(angles_in)

    angles_out = arr_lst_to_padded_batch(angles_out)

    angles = torch.cat((angles_in, angles_out), dim=-1).contiguous()

    

    return (atom_x, bond_x, sc_x, sc_m_x, eucl_dists, graph_dists, angles, mask, 

            bond_idx, sc_idx, angles_idx, sc_types), sc_vals
# create dataloaders and fastai DataBunch

set_seed(100)

batch_size = 20

train_ds = MoleculeDataset(

    train_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,

    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist

)

val_ds   = MoleculeDataset(

    val_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,

    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist

)

test_ds  = MoleculeDataset(

    test_mol_ids, test_gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,

    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist

)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8)

val_dl   = DataLoader(val_ds, batch_size, num_workers=8)

test_dl  = DeviceDataLoader.create(

    test_ds, batch_size, num_workers=8,

    collate_fn=partial(collate_parallel_fn, test=True)

)

db = DataBunch(train_dl, val_dl, collate_fn=collate_parallel_fn)

db.test_dl = test_dl
def reshape_targs(targs, mask_val=BATCH_PAD_VAL):

    targs = targs.view(-1, targs.size(-1))

    return targs[targs[:,0]!=mask_val]



def group_mean_log_mae(y_true, y_pred, types, sc_mean=0, sc_std=1):

    def proc(x): 

        if isinstance(x, torch.Tensor): return x.cpu().numpy().ravel() 

    y_true, y_pred, types = proc(y_true), proc(y_pred), proc(types)

    y_true = sc_mean + y_true * sc_std

    y_pred = sc_mean + y_pred * sc_std

    maes = pd.Series(y_true - y_pred).abs().groupby(types).mean()

    gmlmae = np.log(maes).mean()

    return gmlmae

        

def contribs_rmse_loss(preds, targs):

    """

    Returns the sum of RMSEs for each sc contribution and total sc value.

    """

    targs = reshape_targs(targs)

    return torch.mean((preds - targs) ** 2, dim=0).sqrt().sum()



def rmse(preds, targs):

    targs = reshape_targs(targs)

    return torch.sqrt(F.mse_loss(preds[:,-1], targs[:,-1]))



def mae(preds, targs):

    targs = reshape_targs(targs)

    return torch.abs(preds[:,-1] - targs[:,-1]).mean()



class GroupMeanLogMAE(Callback):

    """Callback to repeort the group mean log MAE during taining."""

    _order = -20 # Needs to run before the recorder



    def __init__(self, learn, snapshot_ensemble=False, **kwargs):

        self.learn = learn

        self.snapshot_ensemble = snapshot_ensemble



    def on_train_begin(self, **kwargs):

        metric_names = ['group_mean_log_mae']

        if self.snapshot_ensemble: metric_names += ['group_mean_log_mae_es']

        self.learn.recorder.add_metric_names(metric_names)

        if self.snapshot_ensemble: self.val_preds = []



    def on_epoch_begin(self, **kwargs):

        self.sc_types, self.output, self.target = [], [], []



    def on_batch_end(self, last_target, last_output, last_input, train,

                     **kwargs):

        if not train:

            sc_types = last_input[-1].view(-1)

            mask = sc_types != BATCH_PAD_VAL

            self.sc_types.append(sc_types[mask])

            self.output.append(last_output[:,-1])

            self.target.append(reshape_targs(last_target)[:,-1])



    def on_epoch_end(self, epoch, last_metrics, **kwargs):

        if (len(self.sc_types) > 0) and (len(self.output) > 0):

            sc_types = torch.cat(self.sc_types)

            preds = torch.cat(self.output)

            target = torch.cat(self.target)

            metrics = [group_mean_log_mae(preds, target, sc_types, SC_MEAN, SC_STD)]



            if self.snapshot_ensemble:

                self.val_preds.append(preds.view(-1,1))

                preds_se = torch.cat(self.val_preds, dim=1).mean(dim=1)

                metrics += [group_mean_log_mae(preds_se, target, sc_types, SC_MEAN, SC_STD)]

            return add_metrics(last_metrics, metrics)
# set up model

set_seed(100)

wd, d_model = 1e-2, 512

enn_args = dict(layers=3*[d_model], dropout=3*[0.0], layer_norm=True)

ann_args = dict(layers=1*[d_model], dropout=1*[0.0], layer_norm=True,

                out_act=nn.Tanh())

model = Transformer(

    N_ATOM_FEATURES, N_BOND_FEATURES, N_SC_EDGE_FEATURES,

    N_SC_MOL_FEATURES, N=6, d_model=d_model, d_ff=d_model*4,

    d_ff_contrib=d_model//4, h=8, dropout=0.0, 

    kernel_sz=min(128, d_model), enn_args=enn_args, ann_args=ann_args

)
callback_fns = [

    GroupMeanLogMAE,

    partial(SaveModelCallback, every='improvement', mode='min',

            monitor='group_mean_log_mae', name=model_str)

]

learn = Learner(db, model, metrics=[rmse, mae], callback_fns=callback_fns,

                wd=wd, loss_func=contribs_rmse_loss)
learn.lr_find(start_lr=1e-7, end_lr=1.0, num_it=100, stop_div=True)

learn.recorder.plot()
learn.fit_one_cycle(15, max_lr=1e-3)

learn.recorder.plot_losses(skip_start=500)
# make predictions

test_contrib_preds = learn.get_preds(DatasetType.Test)

test_preds = test_contrib_preds[0][:,-1].detach().numpy() * SC_STD + SC_MEAN



# store results

submit = pd.read_csv(RAW_DATA_PATH + 'sample_submission.csv')

submit['scalar_coupling_constant'] = test_preds

submit.to_csv(f'{model_str}-submission.csv', index=False)

submit.head()