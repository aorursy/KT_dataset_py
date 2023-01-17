!nvcc --version
# !pip install dgl           # For CPU Build
!pip install dgl-cu101     # For CUDA 10.1 Build
# !pip install dgl-cu90      # For CUDA 9.0 Build
# !pip install dgl-cu92      # For CUDA 9.2 Build
# !pip install dgl-cu100     # For CUDA 10.0 Build
# !pip install dgl-cu101     # For CUDA 10.1 Build
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from dgl import DGLGraph

genre = ['Action','Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',\
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',\
          'Thriller', 'War', 'Western']

def load_data(fold,fformat):
    reviews = pd.read_csv(f'../input/movielens-100k-dataset/ml-100k/u{fold}.{fformat}', sep="\t", header=None)
    reviews.columns = ['user id', 'movie id', 'rating', 'timestamp']

    info = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    info.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', *genre]
    
    users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user', sep="|", encoding='latin-1', header=None)
    users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    
    enc = LabelEncoder()
    users.occupation = enc.fit_transform(users.occupation)
    users.gender = enc.fit_transform(users.gender)
    users["zip code"] = enc.fit_transform(users["zip code"])

    ml = pd.merge(pd.merge(info,reviews),users)
    movie_features = ["movie id", *genre]
    #print(users)
    ux = torch.tensor(ml[users.columns].values)
    mx = torch.tensor(ml[movie_features].values)
    y = torch.tensor(ml.rating.values)
    
    ug = DGLGraph().to(torch.device('cuda:0'))
    ug.add_nodes(5)
    ug.add_edges(*fc(5))

    mg = DGLGraph().to(torch.device('cuda:0'))
    mg.add_nodes(19)
    mg.add_edges(*fc(19))

    rg = DGLGraph().to(torch.device('cuda:0'))
    rg.add_nodes(2)
    rg.add_edges(*fc(2))

    return ug, mg, rg, ux, mx, y


def load_data1M():
    reviews = pd.read_csv('../input/movielens-1m/ml-1m/ratings.dat', delimiter='::', engine='python', header = None)
    reviews.columns = ['user id', 'movie id', 'rating', 'timestamp']

    info = pd.read_csv('../input/movielens-1m/ml-1m/movies.dat', delimiter='::', engine='python', header = None)
    info.columns = ['movie id', 'movie title', 'genre']
    s = info.genre.str.split('|',expand=True)
    x = pd.DataFrame.from_dict({k:np.zeros(len(s)) for k in genre})
    for i in s.columns:
        d = pd.get_dummies(s[i])
        x[d.columns] += d
    info.drop("genre",1)
    info[genre] = x

    users = pd.read_csv('../input/movielens-1m/ml-1m/users.dat', delimiter='::', engine='python', header = None)
    users.columns = ['user id', 'gender', 'age', 'occupation', 'zip code']

    enc = LabelEncoder()
    users.occupation = enc.fit_transform(users.occupation)
    users.gender = enc.fit_transform(users.gender)
    users["zip code"] = enc.fit_transform(users["zip code"])

    ml = pd.merge(pd.merge(info,reviews),users)
    movie_features = ["movie id", *genre]
    #print(users)
    ux = torch.tensor(ml[users.columns].values)
    mx = torch.tensor(ml[genre].values)
    y = torch.tensor(ml.rating.values)
import numpy as np

def fc(nn, labels=None):
    a = list(range(nn))*nn
    b = sorted(list(range(nn)))*(nn-1)
    for i in range(nn):
        a.pop(nn*i)
    a,b = np.array(a),np.array(b)
    if labels is not None:
        a,b = np.array(labels)[a], np.array(labels)[b]
    return a,b
import networkx as nx
import dgl

import matplotlib.pyplot as plt

ug, mg, rg, ux, mx, y = load_data(1,"base")

def export_user():
    plt.figure(figsize=(8, 8))
    mapping = {0:"occupation",1:"age",2:"gender",3:"zip code",4:"user id"}
    F = nx.relabel_nodes(ug.to_networkx(), mapping)
    nx.draw(F, with_labels=True,node_size=3000,font_size=10,font_family='Tahoma',node_color="#f8f8f7", edgecolors='#000000')
    plt.savefig('../working/user.png')
    plt.show()
    
def export_movie():
    plt.figure(figsize=(8, 8))
    mapping = {i:genre[i] for i in range(18)}
    mapping[18] = "movie id"
    H = nx.relabel_nodes(mg.to_networkx(), mapping)
    nx.draw(H, with_labels=True,node_size=3000,font_size=10,font_family='Tahoma',node_color="#f8f8f7", edgecolors='#000000')
    plt.savefig('../working/movie.png')
    plt.show()

def export_rating():
    plt.figure(figsize=(8, 8))
    mapping = {0:"user", 1:"movie"}
    H = nx.relabel_nodes(rg.to_networkx(), mapping)
    nx.draw(H, with_labels=True,node_size=3000,font_size=10,font_family='Tahoma',node_color="#f8f8f7", edgecolors='#000000')
    plt.savefig('../working/rating.png')
    plt.show()

export_user()
export_movie()
export_rating()
import torch.nn as nn
import torch.nn.functional as f
import dgl.function as fn
import dgl

def gcn_msg(edges):
    return {'m': edges.src['h']}

def gcn_reduce(nodes):
    return {'h': nodes.data['h'], 'ms': torch.sum(nodes.mailbox['m'], dim=1)}

def create_graph(g, feature):
    g.ndata['h'] = feature
    return g

class UserLayer(nn.Module):
    def __init__(self, ni, no):
        super(UserLayer, self).__init__()
        self.n_nodes = 5
        self.fc = nn.Linear(ni*self.n_nodes, no)
        
        self.node = nn.GRUCell(ni*self.n_nodes, ni*self.n_nodes)
        self.id = nn.Embedding(943,ni)
        self.age = nn.Embedding(74,ni)
        self.gender = nn.Embedding(2,ni)
        self.occupation = nn.Embedding(21,ni)
        self.zipcode = nn.Embedding(794,ni)
        
    def forward(self, g, feature):
        bs = feature.shape[0]
        with g.local_scope():
            features = self.embed(feature.long())
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(1):#6): #update 3 times
                g.update_all(gcn_msg, gcn_reduce)
                x = g.ndata['ms'].reshape(bs,-1)
                h = g.ndata['h'].reshape(bs,-1)
                g.ndata['h'] = self.node(x,h).reshape(bs*self.n_nodes,-1) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)
    
    def embed(self, feature):
        id = self.id(feature[:,0].unsqueeze(1))#f.one_hot(feature[:,0])
        age = self.age(feature[:,1].unsqueeze(1))#f.one_hot(feature[:,0])
        gender = self.gender(feature[:,2].unsqueeze(1))#f.one_hot(feature[:,1])
        occupation = self.occupation(feature[:,3].unsqueeze(1))#f.one_hot(feature[:,2])
        zipcode = self.zipcode(feature[:,4].unsqueeze(1))#f.one_hot(feature[:,2])
        return torch.cat([id,age,gender,occupation,zipcode],1)
    
class MovieLayer(nn.Module):
    def __init__(self, ni, no):
        super(MovieLayer, self).__init__()
        self.n_nodes = 19
        self.fc = nn.Linear(ni*self.n_nodes, no)
        self.hidden = nn.Linear(ni,ni)
        self.update = nn.Linear(ni,ni)
        self.node = nn.GRUCell(ni*self.n_nodes, ni*self.n_nodes)
        self.embedding = nn.ModuleList([nn.Embedding(1682,ni)])
        for i in range(self.n_nodes):
            self.embedding.append(nn.Embedding(2,ni))
        
    def forward(self, g, feature):
        bs = feature.shape[0]
        with g.local_scope():
            features = self.embed(feature.long())
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(1):#18*2): #update 18 times
                g.update_all(gcn_msg, gcn_reduce)
                x = g.ndata['ms'].reshape(bs,-1)
                h = g.ndata['h'].reshape(bs,-1)
                g.ndata['h'] = self.node(x,h).reshape(bs*self.n_nodes,-1) #rnn
                #g.ndata['h'] = torch.tanh(self.hidden(g.ndata['h'])+self.update(g.ndata['ms'])) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)
    
    def embed(self, feature):
        return torch.cat([self.embedding[i](feature[:,i].unsqueeze(1)) for i in range(self.n_nodes)],1)
    
class UserNet(nn.Module):
    def __init__(self, ni, nf, no):
        super(UserNet, self).__init__()
        self.layer1 = UserLayer(ni, nf)
        self.layer2 = nn.Linear(nf, no)
        
    def forward(self, g, features):
        x = torch.relu(self.layer1(g, features))
        x = self.layer2(x)
        return x

class MovieNet(nn.Module):
    def __init__(self, ni, nf, no):
        super(MovieNet, self).__init__()
        self.layer1 = MovieLayer(ni, nf)
        self.layer2 = nn.Linear(nf, no)
        
    def forward(self, g, features):
        x = f.relu(self.layer1(g, features))
        x = self.layer2(x)
        return x
class RatingLayer(nn.Module):
    def __init__(self, ni, no):
        super(RatingLayer, self).__init__()
        self.fc = nn.Linear(ni*2, no)
        self.node = nn.GRUCell(ni*2, ni*2)
    def forward(self, g, features):
        bs = features.shape[0]
        with g.local_scope():
            g = dgl.batch([create_graph(g,feature) for feature in features])
            for i in range(1): #update 2 times
                g.update_all(gcn_msg, gcn_reduce)
                x = g.ndata['ms'].reshape(bs,-1)
                h = g.ndata['h'].reshape(bs,-1)
                g.ndata['h'] = self.node(x,h).reshape(bs*2,-1)
            #g.ndata['h'] = torch.tanh(self.hidden(g.ndata['h'])+self.update(g.ndata['ms'])) #rnn
            h = g.ndata['h'].reshape(bs,-1)
            return self.fc(h)

        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        
class RatingNet(nn.Module):
    def __init__(self, ni, nf, nir, nfr, no):
        super(RatingNet, self).__init__()
        self.mnet = MovieNet(ni, nf, nir)
        self.unet = UserNet(ni, nf, nir)
        self.layer1 = RatingLayer(nir, nfr)
        self.layer2 = MLP(get_layers(nfr,512,no,128))
        self.apply(init_weights)
    def forward(self, ug, mg, rg, ufeatures, mfeatures):
        uxt = self.unet(ug, ufeatures)
        mxt = self.mnet(mg, mfeatures)
        rx = torch.cat([mxt.unsqueeze(1),uxt.unsqueeze(1)],1)
        x = f.relu(self.layer1(rg,rx))
        x = self.layer2(x)
        return x
def get_layers(start, hs, end, step):
    lse = [*list(range(hs, end, -step)), end]
    return list(zip([start,*lse[:]], [*lse[:], end]))[:-1]

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.LeakyReLU()) for n in layers])
    def forward(self, x):
        return self.model(x)

class umMLP(nn.Module):
    def __init__(self, layers, ne):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.LeakyReLU()) for n in layers])
        self.embedding = nn.ModuleList()
        for i in range(18):
            self.embedding.append(nn.Embedding(2,ne))
        self.embedding.append(nn.Embedding(74,ne))
        self.embedding.append(nn.Embedding(2,ne))
        self.embedding.append(nn.Embedding(21,ne))
        self.apply(init_weights)
    def forward(self, ux, mx):
        u = torch.cat([self.embedding[i](ux[:,i-18].unsqueeze(1)) for i in range(18,18+3)],1)
        m = torch.cat([self.embedding[i](mx[:,i].unsqueeze(1)) for i in range(18)],1)
        x = torch.cat([u,m],1).flatten(1)
        return self.model(x)
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pickle
import math

def evaluate(net, vdl):
    eloss = 0
    net.eval()
    with torch.no_grad():
        for i,(u,m,label) in enumerate(vdl):
            out = net(ug, mg, rg, u.cuda(), m.cuda())
            loss = torch.sqrt(f.mse_loss(out, label.cuda().float()))
            eloss += loss.item()
    return eloss/i

for FOLD in range(1,5+1):
    ug, mg, rg, ux, mx, y = load_data(FOLD,"base")
    tdl = DataLoader(TensorDataset(ux,mx,y), batch_size=512, shuffle=True)
    ug, mg, rg, ux, mx, y = load_data(FOLD,"test")
    vdl = DataLoader(TensorDataset(ux,mx,y), batch_size=512, shuffle=True)

    net = RatingNet(256,256,1024,2048,1).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, amsgrad=True)

    pb = tqdm(range(100))

    tl = []
    vl = []

    for epoch in pb:
        net.train()
        eloss = 0
        for i,(u,m,label) in enumerate(tdl):
            out = net(ug, mg, rg, u.cuda(), m.cuda())
            loss = f.mse_loss(out, label.cuda().float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            eloss += math.sqrt(loss.item())
            pb.set_description(f"{epoch}| {eloss/(i+1)} | {i}")

        vloss = evaluate(net, vdl)
        eloss /= i

        tl.append(eloss)
        vl.append(vloss)

        print(f"{epoch}| {eloss} | {vloss}")
def evaluate(net, vdl):
    eloss = 0
    net.eval()
    with torch.no_grad():
        for i,(u,m,label) in enumerate(vdl):
            out = net(u.cuda(), m.cuda())
            loss = torch.sqrt(crit(out, label.cuda().float()))
            eloss += loss.item()
    return eloss/i

for FOLD in range(1,5+1):
    ux, mx, y = load_data(FOLD,"base")
    tdl = DataLoader(TensorDataset(ux,mx,y), batch_size=512, shuffle=True)
    ux, mx, y = load_data(FOLD,"test")
    vdl = DataLoader(TensorDataset(ux,mx,y), batch_size=512, shuffle=True)

    net = net = umMLP(get_layers(128*21,512,1,128),128).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    pb = tqdm(range(10))

    tl = []
    vl = []

    for epoch in pb:
        net.train()
        eloss = 0
        for i,(u,m,label) in enumerate(tdl):
            out = net(u.cuda(), m.cuda())
            loss = crit(out, label.cuda().float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            eloss += math.sqrt(loss.item())

            pb.set_description(f"{epoch}| {eloss/(i+1)} | {i}")

        vloss = evaluate(net, vdl)
        eloss /= i

        tl.append(eloss)
        vl.append(vloss)

        print(f"{epoch}| {eloss} | {vloss}")
# from dgl.data import citation_graph as citegrh
# import networkx as nx
# import torch

# def load_cora_data():
#     data = citegrh.load_cora()
#     features = torch.FloatTensor(data.features)
#     labels = torch.LongTensor(data.labels)
#     train_mask = torch.BoolTensor(data.train_mask)
#     test_mask = torch.BoolTensor(data.test_mask)
#     g = DGLGraph(data.graph)
#     return g, features, labels, train_mask, test_mask
# g, features, labels, train_mask, test_mask = load_cora_data()
# import torch.nn as nn
# import dgl.function as fn

# gcn_msg = fn.copy_src(src='h', out='m')
# gcn_reduce = fn.sum(msg='m', out='h')

# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
        
#     def forward(self, g, feature):
#         with g.local_scope():
#             g.ndata['h'] = feature
#             g.update_all(gcn_msg, gcn_reduce)
#             h = g.ndata['h']
#             return self.linear(h)
        
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.layer1 = GCNLayer(1433, 16)
#         self.layer2 = GCNLayer(16, 7)

#     def forward(self, g, features):
#         x = f.relu(self.layer1(g, features))
#         x = self.layer2(g, x)
#         return x
    
# net = Net()
# print(net)
# !pip install torch-scatter
# !pip install torch-sparse
# !pip install torch-cluster
# !pip install torch-spline-conv
# !pip install torch_geometric
# import pandas as pd
# import pickle

# df = pd.read_csv('../input/recsys-challenge-2015/yoochoose-clicks.dat', header=None)
# df.columns=['session_id','timestamp','item_id','category']
# df['valid_session'] = df.session_id.map(df.groupby('session_id')['item_id'].size() > 2)
# df = df.loc[df.valid_session].drop('valid_session',axis=1)

# buy_df = pd.read_csv('../input/recsys-challenge-2015/yoochoose-buys.dat', header=None)
# buy_df.columns=['session_id','timestamp','item_id','price','quantity']
# buy_df.head()

# df['label'] = df.session_id.isin(buy_df.session_id)

# !wget "https://snap.stanford.edu/data/amazon0302.txt.gz"
# import gzip
# import shutil
# with gzip.open('./amazon0302.txt.gz', 'rb') as f_in:
#     with open('./amazon0302.txt', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
# import numpy as np
# import torch
# from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
# from torch_sparse import coalesce
# df = pd.read_csv('./amazon0302.txt', sep='\t', header=None, skiprows=4, dtype=np.int64)
# edge_index = torch.from_numpy(df.values).t()
# num_nodes = edge_index.max().item() + 1

# radius = 2
# for s, group in df.groupby(0):
#     group = group.reset_index(drop=True)
#     neighbors = [*group.values.transpose()[-1]]
#     for n in group.values.transpose()[-1]:
#         neighbors.extend(df.loc[df[0] == n][1])
#         print(n, df.loc[df[0] == n][1])
#     break
# set(neighbors)
# data = Data(edge_index=edge_index, num_nodes=num_nodes)
# from torch_geometric.data import InMemoryDataset
# from tqdm import tqdm
# import torch

# class AmazonDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(AmazonDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return []
#     @property
#     def processed_file_names(self):
#         return ['amazon0302.dataset']
#     def download(self):
#         pass
#     def process(self):
#         data_list = []
#         df = pd.read_csv('./amazon0302.txt', sep='\t', header=None, skiprows=4, dtype=np.int64)
#         edge_index = torch.from_numpy(df.values).t()
#         num_nodes = edge_index.max().item() + 1
#         data_list = [Data(edge_index=edge_index, num_nodes=num_nodes)]
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])
# dataset = AmazonDataset(root='.')
# from torch_geometric.data import DataLoader

# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)
# dataset.num_classes
