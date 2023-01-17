import numpy as np
import pandas as pd
import torch 
from torch import nn
import torch.nn.functional as F
from torch import optim
from IPython.core.debugger import set_trace
df = pd.read_csv('../input/adult.csv')
train_df, valid_df = df[:-2000].copy(),df[-2000:].copy()

train_df.head()
train_df.isnull().sum() # check NaN
full_col = list(train_df.columns)
full_col.remove('>=50k') # all the independent data
full_col
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cat_names # categorical variables
cont_names = [col for col in full_col if col not in cat_names]
cont_names # continuous variables
for n in cat_names:
    train_df[n] = train_df[n].astype('category').cat.as_ordered()
train_df.dtypes # data types after categorifying
train_df['occupation'].cat.categories
train_df['occupation'].cat.codes
train_df['education-num'].median()
train_df['education-num'].head()
for n in cont_names:
    if pd.isnull(train_df[n]).sum():
        filler = train_df[n].median()
        train_df[n] = train_df[n].fillna(filler)
train_df['education-num'].head()
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
dep_var = '>=50k' # dependent variable
if not is_numeric_dtype(df[dep_var]): train_df[dep_var] = train_df[dep_var].cat.codes
y = torch.tensor(train_df[dep_var].values)
y
train_df['sex'].cat.codes.values
if cat_names and len(cat_names) >= 1: # categorical data
    cats = np.stack([c.cat.codes.values for n,c in train_df[cat_names].items()], 1) + 1
cats.shape
cats = torch.LongTensor(cats.astype(np.int64))
cont_names # continuous data
if cont_names and len(cont_names) >= 1:
    conts = np.stack([c.astype('float32').values for n,c in train_df[cont_names].items()], 1)
    means, stds = (conts.mean(0), conts.std(0))
    conts = (conts - means[None]) / stds[None]
    stats = means,stds
conts = torch.FloatTensor(conts)
bs = 64 #
xb_cont = conts[0:bs]
xb_cat = cats[0:bs]
yb = y[:bs]
cat_szs = [len(train_df[n].cat.categories)+1 for n in cat_names]
emb_szs = [(c, min(50, (c+1)//2)) for c in cat_szs]
emb_szs
def bn_drop_lin(n_in, n_out, bn, p, actn):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers
class TabularModel(nn.Module):
    "Basic model for tabular data"
    
    def __init__(self, emb_szs, n_cont, out_sz, layers, drops, 
                 emb_drop, use_bn, is_reg, is_multi):
        super().__init__()
        
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+drops,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont) # why batch norm here ??
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        return x.squeeze()
model = TabularModel(emb_szs, len(cont_names), 2, [200,100], [0.001,0.01], emb_drop=0.04, is_reg=False,is_multi=True, use_bn=True)
model
model(xb_cat, xb_cont) # Test if model works
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds==yb).float().mean()
loss_func = F.cross_entropy 
loss_func(model(xb_cat, xb_cont), yb)
accuracy(model(xb_cat, xb_cont), yb)
opt = optim.SGD(model.parameters(), lr=1e-2)
epochs = 15

n,c = cats.shape # number of sample and categorical variables
for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        start_i = i*bs
        end_i = start_i+bs
        xb_cont = conts[start_i:end_i]
        xb_cat = cats[start_i:end_i]
        yb = y[start_i:end_i]
        pred = model(xb_cat, xb_cont)
        loss = loss_func(model(xb_cat, xb_cont), yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
loss_func(model(xb_cat, xb_cont), yb)
accuracy(model(xb_cat, xb_cont), yb)