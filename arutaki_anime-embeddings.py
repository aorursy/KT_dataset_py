from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
anime = pd.read_csv("../input/anime-recommendations-database/anime.csv")
rating = pd.read_csv("../input/anime-recommendations-database/rating.csv")
anime = anime.sort_values('anime_id')
anime
rated = rating[rating.rating >= 0]
mean = rated["rating"].mean()
rated = rated[rated.rating > mean]
test = rated.groupby("user_id").filter(lambda x:len(x) >= 10)
test = test.groupby("user_id").filter(lambda x:len(x) <= 20)

test
cross = pd.crosstab(index = test.user_id, columns = test.anime_id)
cross
cross_int = cross.astype(int)
cooc = cross_int.T.dot(cross_int)
cooc
once = [cooc.columns[i] for i in range(len(cooc.columns)) if cooc.iloc[i, i] == 1]
cooc = cooc.drop(columns = once)
cooc = cooc.drop(index = once)
cooc
class AnimeDataset:
    def __init__(self, coocc_matrix, anime_df):
        self.coocc = coocc_matrix
        self.anime = anime_df
        
        self.good_id = list(self.coocc.columns)        
        self.namelen = len(self.good_id)
        
        self.name = [self.anime.name[self.anime.anime_id == i].values[0] for i in self.good_id if len(self.anime.name[self.anime.anime_id == i].values) != 0]
        self.newid = list(range(self.namelen))
        self.id2name = dict(zip(self.newid, self.name))
        
        newcol = list(range(self.namelen))
        
        self.coocc.columns = newcol
        self.coocc.index = newcol
        
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        
        for i in range(self.namelen):
            for j in range(self.namelen):
                if i != j and self.coocc.loc[i, j] != 0:
                    self._i_idx.append(i)
                    self._j_idx.append(j)
                    self._xij.append(self.coocc.loc[i, j])
        
        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()
        
    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
dataset = AnimeDataset(cooc, anime)
class AnimeGlove(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(AnimeGlove, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)
        
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
        
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        
        return x
EMBED_DIM = 100
NAME_LEN = len(cooc)
model = AnimeGlove(NAME_LEN, EMBED_DIM)
model.cuda()
def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.cuda()  

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).cuda()

optimizer = optim.Adagrad(model.parameters(), lr=0.05)
N_EPOCHS = 100
BATCH_SIZE = 2048
X_MAX = 100
ALPHA = 0.75
n_batches = int(len(dataset._xij) / BATCH_SIZE)
loss_values = list()
for e in range(1, N_EPOCHS+1):
    batch_i = 0

    for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):

        batch_i += 1

        optimizer.zero_grad()

        outputs = model(i_idx, j_idx)

        weights_x = weight_func(x_ij, X_MAX, ALPHA)

        loss = wmse_loss(weights_x, outputs, torch.log(x_ij))

        loss.backward()

        optimizer.step()

        loss_values.append(loss.item())

        if batch_i % 100 == 0:
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))  
    
    print("Saving model...")
    torch.save(model.state_dict(), "anime.pt")
plt.plot(loss_values)
emb_i = model.wi.weight.cpu().data.numpy()
emb_j = model.wj.weight.cpu().data.numpy()
emb = emb_i + emb_j
top_k = 300
tsne = TSNE(metric='cosine', random_state=123)
embed_tsne = tsne.fit_transform(emb[:top_k, :])
fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(top_k):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(dataset.id2name[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
from gensim.models import Word2Vec