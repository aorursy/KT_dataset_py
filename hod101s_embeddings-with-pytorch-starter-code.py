import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()
df.head()
df.shape
df.info()
pd.DataFrame(data=[[col,len(df[col].unique())] for col in df.columns],columns=['Feature','Unique Items']).style.background_gradient()
df['YearsSinceBuilt'] = datetime.datetime.now().year - df.YearBuilt
df.drop('YearBuilt',axis=1,inplace=True)
df.head()
df.columns
cat_features = ['MSSubClass', 'MSZoning','Street','LotShape', ]
lblEncoders = {}
for feature in cat_features:
    lblEncoders[feature] = LabelEncoder()
    df[feature] = lblEncoders[feature].fit_transform(df[feature])
catTensor = np.stack([df[col] for col in cat_features],1)
catTensor = torch.tensor(catTensor,dtype=torch.int64)
catTensor
cont_features = [col for col in df.columns if col not in cat_features and col!= 'SalePrice']
cont_features 
contTensor = np.stack([df[col] for col in cont_features],axis=1)
contTensor = torch.tensor(contTensor,dtype=torch.float)
contTensor
y = torch.tensor(df['SalePrice'].values,dtype=torch.float).reshape(-1,1)
y
cat_dims = [len(df[col].unique()) for col in cat_features]
embedding_dims = [(dim,min(50,(dim+1)//2)) for dim in cat_dims]
embedding_dims
embed_repr = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])
embed_repr
embed_values = []
for i,e in enumerate(embed_repr):
    embed_values.append(e(catTensor[:,i]))
embed_values = torch.cat(embed_values,1)
embed_values
embed_values = nn.Dropout(.4)(embed_values)
embed_values
class Model(nn.Module):
    def __init__(self, embedding_dim, n_cont, out_sz, layers, drop=0.5):
        super().__init__()
        self.embed_repr = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dims])
        self.embed_dropout = nn.Dropout(drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((val[1] for val in embedding_dim))
        n_in = n_cont + n_emb
        
        for layer in layers:
            layerlist.append(nn.Linear(n_in,layer))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(layer))
            layerlist.append(nn.Dropout(drop))
            n_in = layer
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        self.layers = nn.Sequential(*layerlist)
        
    def forward(self, cat,cont):
        embeddings = []
        for i,e in enumerate(self.embed_repr):
            embeddings.append(e(cat[:,i]))
        x = torch.cat(embeddings,1)
        x = self.embed_dropout(x)
        x_cont = self.bn_cont(cont)
        x = torch.cat([x,x_cont],1)
        x = self.layers(x)
        return x
torch.manual_seed(100)
model = Model(embedding_dims, len(cont_features), 1, [100,50], drop = .4)
model
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =0.01)
batch_size = len(df)
test_size = int(batch_size*.15)
train_cat = catTensor[:batch_size-test_size]
test_cat = catTensor[batch_size-test_size:batch_size]
train_cont = contTensor[:batch_size-test_size]
test_cont = contTensor[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]
len(test_cat)
epochs = 5000
losses = []
for i in range(epochs):
    i += 1
    y_pred = model.forward(train_cat,train_cont)
    loss = torch.sqrt(loss_function(y_pred,y_train))
    losses.append(loss)
    if i%50 == 0:
        print(f"Epoch {i} : {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')
with torch.no_grad():
    y_pred=model(test_cat,test_cont)
    loss=torch.sqrt(loss_function(y_pred,y_test))
print('RMSE: {}'.format(loss))