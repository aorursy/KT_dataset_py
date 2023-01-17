import torch

print(torch.__version__)

print(torch.version.cuda)
!pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html

!pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html

!pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html

!pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html

!pip install torch-geometric
import warnings

warnings.filterwarnings('ignore')



import os

import shutil



#the basics

import pandas as pd, numpy as np, seaborn as sns

import math, json

import matplotlib.pyplot as plt

import matplotlib.colors as mc

from matplotlib import cm

import seaborn as sns

import colorsys

from tqdm import tqdm



#for model evaluation

from sklearn.model_selection import train_test_split, KFold



import torch.nn as nn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
!conda install -y -c bioconda forgi

!conda install -y -c bioconda viennarna



import forgi.graph.bulge_graph as fgb

import forgi.visual.mplotlib as fvm

import forgi.threedee.utilities.vector as ftuv

import forgi



import RNA
def seed_all(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    

seed_all()
class config:

    learning_rate = 0.001

    K = 1 # number of aggregation loop (also means number of GCN layers)

    gcn_agg = 'mean' # aggregator function: mean, conv, lstm, pooling

    filter_noise = True

    seed = 1234

    noise_threshold = 1
def get_couples(structure):

    """

    For each closing parenthesis, I find the matching opening one and store their index in the couples list.

    The assigned list is used to keep track of the assigned opening parenthesis

    """

    opened = [idx for idx, i in enumerate(structure) if i == '(']

    closed = [idx for idx, i in enumerate(structure) if i == ')']



    assert len(opened) == len(closed)





    assigned = []

    couples = []



    for close_idx in closed:

        for open_idx in opened:

            if open_idx < close_idx:

                if open_idx not in assigned:

                    candidate = open_idx

            else:

                break

        assigned.append(candidate)

        couples.append([candidate, close_idx])

        assigned.append(close_idx)

        couples.append([close_idx, candidate])

        

    assert len(couples) == 2*len(opened)

    

    return couples
def build_matrix(couples,size):

    mat = np.zeros((size, size))

    

    for i in range(size):  # neigbouring bases are linked as well

        if i < size - 1:

            mat[i, i + 1] = 1

        if i > 0:

            mat[i, i - 1] = 1

    

    for i, j in couples:

        mat[i, j] = 2

        mat[j, i] = 2

        

    return mat
def seq2nodes(sequence,loops,structures):

    type_dict={'A':0,'G':1,'U':2,'C':3}

    loop_dict={'S':0,'M':1,'I':2,'B':3,'H':4,'E':5,'X':6}

    struct_dict={'.':0,'(':1,')':2}

    # 4 types, 7 structural types

    nodes=np.zeros((len(sequence),4+7+3))

    for i,s in enumerate(sequence):

        nodes[i,type_dict[s]]=1

    for i,s in enumerate(loops):

        nodes[i,4+loop_dict[s]]=1

    for i,s in enumerate(structures):

        nodes[i,11+struct_dict[s]]=1

    return nodes
all_data=pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

all_data.head(5)
idx = 0

id_=all_data.iloc[idx].id

sequence = all_data.iloc[idx].sequence

structure = all_data.iloc[idx].structure

loops=all_data.iloc[idx].predicted_loop_type

reactivity = all_data.iloc[idx].reactivity

bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

fig = plt.figure(figsize=(6, 6))



fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None})

plt.show()
matrix=build_matrix(get_couples(structure),len(sequence))

bpps_dir='../input/stanford-covid-vaccine/bpps/'

bpps=np.load(bpps_dir+id_+'.npy')

edge_index=np.stack(np.where((matrix+bpps)>0))

#adjacents=np.stack(np.where(matrix==1))

#couples=np.stack(np.where(matrix==2))

#probs=np.stack(np.where(bpps>0))

#edge_index=np.hstack((adjacents,couples))

# nodes x features

node_attr=seq2nodes(sequence,loops,structure)

edge_attr=np.zeros((edge_index.shape[1],4))

edge_attr[:,0]=(matrix==2)[edge_index[0,:],edge_index[1,:]]

edge_attr[:,1]=(matrix==1)[edge_index[0,:],edge_index[1,:]]

edge_attr[:,2]=(matrix==-1)[edge_index[0,:],edge_index[1,:]]

edge_attr[:,3]=bpps[edge_index[0,:],edge_index[1,:]]
from torch_geometric.data import InMemoryDataset

from torch_geometric.data import Data



class MyOwnDataset(InMemoryDataset):

    def __init__(self, root='',train=True, public=True, ids=None,filter_noise=False,transform=None, pre_transform=None):

        try:

            shutil.rmtree('./'+root)

        except:

            print("doesn't exist")

        self.train=train

        if self.train:

            self.data_dir = '../input/stanford-covid-vaccine/train.json'

        else:

            self.data_dir = '/kaggle/input/stanford-covid-vaccine/test.json'

        self.bpps_dir='../input/stanford-covid-vaccine/bpps/'

        self.df=pd.read_json(self.data_dir,lines=True)

        if filter_noise:

            self.df = self.df[self.df.SN_filter ==1]

        if ids is not None:

            self.df=self.df[self.df['index'].isin(ids)]

        if public:

            self.df=self.df.query("seq_length == 107")

        else:

            self.df=self.df.query("seq_length == 130")

        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']

        

        super(MyOwnDataset, self).__init__(root,transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        

    

    @property

    def raw_file_names(self):

        return []



    @property

    def processed_file_names(self):

        return 'data.pt'



    def download(self):

        pass



    def process(self):

        # Read data into huge `Data` list.

        data_list = []

        for idx in range(len(self.df)):

            structure=self.df['structure'].iloc[idx]

            sequence=self.df['sequence'].iloc[idx]

            loops=self.df['predicted_loop_type'].iloc[idx]

            # 2 x edges

            matrix=build_matrix(get_couples(structure),len(sequence))

            # nodes x features

            id_=self.df['id'].iloc[idx]

            bpps=np.load(self.bpps_dir+id_+'.npy')

            edge_index=np.stack(np.where((matrix)!=0))

            node_attr=seq2nodes(sequence,loops,structure)

            node_attr=np.append(node_attr, bpps.sum(axis=1,keepdims=True), axis=1)

            edge_attr=np.zeros((edge_index.shape[1],4))

            edge_attr[:,0]=(matrix==2)[edge_index[0,:],edge_index[1,:]]

            edge_attr[:,1]=(matrix==1)[edge_index[0,:],edge_index[1,:]]

            edge_attr[:,2]=(matrix==-1)[edge_index[0,:],edge_index[1,:]]

            edge_attr[:,3]=bpps[edge_index[0,:],edge_index[1,:]]

            # targets

            #padded_targets=np.zeros((130,5))

            if self.train:

                targets=np.stack(self.df[self.target_cols].iloc[idx]).T

            else:

                targets=np.zeros((130,5))

            x = torch.from_numpy(node_attr)

            y = torch.from_numpy(targets)

            edge_attr=torch.from_numpy(edge_attr)

            edge_index=torch.tensor(edge_index,dtype=torch.long)

            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y)

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)

            data.train_mask[:68] = 1

            data_list.append(data)



        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
all_ids=np.arange(len(all_data))

np.random.shuffle(all_ids)

train_ids,val_ids=np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])



train_dataset=MyOwnDataset(ids=train_ids, root='train',filter_noise=config.filter_noise)

val_dataset=MyOwnDataset(ids=val_ids, root='val',filter_noise=config.filter_noise)



from torch_geometric.data import DataLoader



train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
import torch

import torch.nn.functional as F

from torch_geometric.nn import GCNConv



class GCNNet(torch.nn.Module):

    def __init__(self, node_feats,channels,out_feats,edge_feats=1):

        super(GCNNet, self).__init__()

        self.conv1 = GCNConv(node_feats, channels)

        self.conv2 = GCNConv(channels, channels)

        self.conv3 = GCNConv(channels, channels)

        self.conv4 = GCNConv(channels, channels)

        self.conv5 = GCNConv(channels, channels)

        self.conv9 = GCNConv(channels, out_feats)



    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv9(x, edge_index)

        return x

    



import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.nn import NNConv, Set2Set    



class MPNNet(torch.nn.Module):

    def __init__(self, node_feats,channels,out_feats,loops=1,edge_feats=1):

        super(MPNNet, self).__init__()

        self.lin0 = torch.nn.Linear(node_feats,channels)

        self.loops=loops

        nn = Sequential(Linear(edge_feats, channels), ReLU(), Linear(channels, channels * channels))

        self.conv = NNConv(channels, channels, nn, aggr='mean')

        self.gru = GRU(channels, channels)



        self.lin1 = torch.nn.Linear(channels, channels)

        self.lin2 = torch.nn.Linear(channels, out_feats)



    def forward(self, data):

        out = F.relu(self.lin0(data.x))

        h = out.unsqueeze(0)



        for i in range(self.loops):

            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))

            out, h = self.gru(m.unsqueeze(0), h)

            out = out.squeeze(0)



        #out = self.set2set(out, data.batch)

        out = F.relu(self.lin1(out))

        out = self.lin2(out)

        return out

node_feats=train_dataset.num_node_features

out_feats=train_dataset.num_classes

edge_feats=train_dataset.num_edge_features





#model = GCNNet(node_feats,256,out_feats,edge_feats=edge_feats).double().to(device)

model = MPNNet(node_feats,128,out_feats,loops=10,edge_feats=edge_feats).double()

print(sum(p.numel() for p in model.parameters()))

#data = dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters())

#loss_fn = nn.MSELoss()
class MCRMSELoss(torch.nn.Module):

    def __init__(self):

        super(MCRMSELoss,self).__init__()



    def forward(self,x,y):

        #columnwise mean

        x=x[:,:3]

        y=y[:,:3]

        msq_error=torch.mean((x-y)**2,0)

        loss=torch.mean(torch.sqrt(msq_error))

        return loss
class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
from torch.nn import MSELoss

import gc

loss_fn = MCRMSELoss()

#loss_fn = MSELoss()



def train(model,optimizer,train_loader):

    model.train()

    train_loss = AverageMeter()

    for batch_idx,data in enumerate(train_loader):# Iterate in batches over the training dataset.

        out = model(data.to(device))  # Perform a single forward pass.

        loss = loss_fn(out[data.train_mask], data.y)  # Compute the loss.

        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.

        optimizer.zero_grad()

        train_loss.update(loss.item())

    return train_loss.avg



def test(model,val_loader):

    model.eval()

    val_loss = AverageMeter()

    for batch_idx,data in enumerate(val_loader):  # Iterate in batches over the training/test dataset.

        out = model(data.to(device))

        loss=loss_fn(out[data.train_mask], data.y)

        val_loss.update(loss.item())  # Compute the loss. # Check against ground-truth labels.

    return val_loss.avg
def train_loop(model,epochs=1):

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)

    train_loss = []

    val_loss = []

    for epoch in range(1, epochs+1):

        train_acc = train(model,optimizer,train_loader)

        val_acc = test(model,val_loader)

        train_loss.append(train_acc)

        val_loss.append(val_acc)

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {val_acc:.4f}')

    return model, train_loss, val_loss
model, train_loss, val_loss = train_loop(model,epochs=100)
def custom_plot_rna(cg, coloring, ax=None):

    '''

    Edited from https://github.com/ViennaRNA/forgi/blob/master/forgi/visual/mplotlib.py

    '''

    RNA.cvar.rna_plot_type = 1

    coords = []

    bp_string = cg.to_dotbracket_string()

    if ax is None:

        ax = plt.gca()

    vrna_coords = RNA.get_xy_coordinates(bp_string)

    

    for i, _ in enumerate(bp_string):

        coord = (vrna_coords.get(i).X, vrna_coords.get(i).Y)

        coords.append(coord)

    coords = np.array(coords)

    

    # Now plot circles

    for i, coord in enumerate(coords):

        if i < len(coloring):

            c = cm.coolwarm(coloring[i])

        else: 

            c = 'grey'

        h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))

        c=colorsys.hls_to_rgb(h,l,s)

        circle = plt.Circle((coord[0], coord[1]),color=c)

        ax.add_artist(circle)



    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),

                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),

               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),

                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))



    width = datalim[1][0] - datalim[0][0]

    height = datalim[1][1] - datalim[0][1]



    ax.set_aspect('equal', 'datalim')

    ax.update_datalim(datalim)

    ax.autoscale_view()

    ax.set_axis_off()



    return (ax, coords)



def plot_structure_with_target_var(idx):

    sequence = all_data.iloc[idx].sequence

    structure = all_data.iloc[idx].structure



    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))

    coloring = all_data.iloc[idx].reactivity

    coloring = [(c-min(all_data.reactivity[idx]))/(max(all_data.reactivity[idx])-min(all_data.reactivity[idx])) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[0])

    ax[0].set_title('reactivity', fontsize=16)



    coloring = all_data.iloc[idx].deg_Mg_pH10

    coloring = [(c-min(all_data.deg_Mg_pH10[idx]))/(max(all_data.deg_Mg_pH10[idx])-min(all_data.deg_Mg_pH10[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[1])

    ax[1].set_title('deg_Mg_pH10', fontsize=16)



    coloring = all_data.iloc[idx].deg_pH10

    coloring = [(c-min(all_data.deg_pH10[idx]))/(max(all_data.deg_pH10[idx])-min(all_data.deg_pH10[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[2])

    ax[2].set_title('deg_pH10', fontsize=16)



    coloring = all_data.iloc[idx].deg_Mg_50C

    coloring = [(c-min(all_data.deg_Mg_50C[idx]))/(max(all_data.deg_Mg_50C[idx])-min(all_data.deg_Mg_50C[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[3])

    ax[3].set_title('deg_Mg_50C', fontsize=16)



    coloring = all_data.iloc[idx].deg_50C

    coloring = [(c-min(all_data.deg_50C[idx]))/(max(all_data.deg_50C[idx])-min(all_data.deg_50C[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[4])

    ax[4].set_title('deg_50C', fontsize=16)



    plt.show()

    

def plot_structure_with_predicted_var(idx):

    sequence = all_data.iloc[idx].sequence

    structure = all_data.iloc[idx].structure

    try:

        data=train_dataset[idx]

    except:

        data=val_dataset[idx]

    preds = model(data.to(device)).detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))



    coloring = preds[:,0].tolist()

    coloring = [(c-min(all_data.reactivity[idx]))/(max(all_data.reactivity[idx])-min(all_data.reactivity[idx])) for c in coloring] 

    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]

    custom_plot_rna(bg, coloring, ax=ax[0])

    ax[0].set_title('reactivity', fontsize=16)



    coloring = preds[:,1].tolist()

    coloring = [(c-min(all_data.deg_Mg_pH10[idx]))/(max(all_data.deg_Mg_pH10[idx])-min(all_data.deg_Mg_pH10[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[1])

    ax[1].set_title('deg_Mg_pH10', fontsize=16)



    coloring = preds[:,2].tolist()

    coloring = [(c-min(all_data.deg_pH10[idx]))/(max(all_data.deg_pH10[idx])-min(all_data.deg_pH10[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[2])

    ax[2].set_title('deg_pH10', fontsize=16)



    coloring = preds[:,3].tolist()

    coloring = [(c-min(all_data.deg_Mg_50C[idx]))/(max(all_data.deg_Mg_50C[idx])-min(all_data.deg_Mg_50C[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[3])

    ax[3].set_title('deg_Mg_50C', fontsize=16)



    coloring = preds[:,4].tolist()

    coloring = [(c-min(all_data.deg_50C[idx]))/(max(all_data.deg_50C[idx])-min(all_data.deg_50C[idx])) for c in coloring] 

    custom_plot_rna(bg, coloring, ax=ax[4])

    ax[4].set_title('deg_50C', fontsize=16)



    plt.show()


#Choose an index to look at

idx=val_ids[0]



plot_structure_with_target_var(5)

plot_structure_with_predicted_var(5)
import matplotlib.pyplot as plt

plt.plot(train_loss,label='train')

plt.plot(val_loss,label='val')

plt.title('Plot training and validation losses')

plt.legend()
import gc

del train_dataset

del train_loader

del val_dataset

del val_loader

gc.collect()
public_leaderboard_dataset=MyOwnDataset(root='public/',train=False,public=True)

private_leaderboard_dataset=MyOwnDataset(root='private/',train=False,public=False)



public_leaderboard_loader = DataLoader(public_leaderboard_dataset, batch_size=4, shuffle=False)

private_leaderboard_loader = DataLoader(private_leaderboard_dataset, batch_size=4, shuffle=False)
def get_preds(pred_loader,public=False):

    model.eval()

    batch_preds=[]

    for batch_idx,data in enumerate(pred_loader):

        out = model(data.to(device))

        if public:

            out=out

        else:

            out=out

        batch_preds.append(out.cpu().detach())

    return batch_preds
public_preds=get_preds(public_leaderboard_loader,public=True)

private_preds=get_preds(private_leaderboard_loader,public=False)
public_preds=torch.cat(public_preds,dim=0).numpy()

private_preds=torch.cat(private_preds,dim=0).numpy()
all_df=pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json',lines=True)

public_df = all_df.query("seq_length == 107")

private_df = all_df.query("seq_length == 130")
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']

preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        sequence=df.sequence.iloc[i]

        single_pred = preds[i*len(sequence):i*len(sequence)+len(sequence),:]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)

preds_df.head(10)
sample_df = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)
submission.head(10)
print(len(submission))