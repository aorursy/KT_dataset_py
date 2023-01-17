# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.

import torch

from torch import nn, optim

from torch.utils.data import DataLoader, TensorDataset,random_split

# from torch.



from pandas.api.types import is_numeric_dtype

from matplotlib import pyplot as plt

from IPython.display import clear_output
def read_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    return train, test



def get_cat_cols(data_df):

    return list(data_df.select_dtypes(include=['category', 'object']))





def get_num_cols(data_df):

    return list(data_df.select_dtypes(exclude=['category', 'object']))
train, test = read_data()

print('train.shape: ', train.shape)

print('test.shape : ', test.shape)



target = 'SalePrice'

ignore_cols = 'Id'

print('target: ', target)
cat_cols, cont_cols = [],[]

for col in train:

    if col == target or col in ignore_cols:

        continue

    if (train[col].dtype == 'int' and train[col].unique().shape[0] > 20) or train[col].dtype == 'float':

        cont_cols.append(col)

    else:

        cat_cols.append(col)



print('cat_cols: {0}, cont_cols: {1}'.format(len(cat_cols), len(cont_cols)))
"""

Fill missing values for categorical and continuous features

"""

class FillMissing():

    cont_cols = []

    col_filler = {}

    

    def __init__(self,cont_cols,cat_cols):

        self.cont_cols = cont_cols

        self.cat_cols = cat_cols

        

    def _fill_cat_numeric(self, df, col):

        if is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(-999)

        else:

            df[col] = df[col].fillna("#na#")

    

    def apply_train(self, df):

        for col in self.cont_cols:

            filler = df[col].median()

            self.col_filler[col] = filler

            df[col] = df[col].fillna(filler)

        for col in self.cat_cols:

            self._fill_cat_numeric(df, col)

        

    def apply_test(self, df):

        for col in self.cont_cols:

            df[col] = df[col].fillna(self.col_filler.get(col))

        for col in self.cat_cols:

            self._fill_cat_numeric(df, col)

 

"""

Normalize continuous featurs and also the target (for regression)

"""

class Normalize():

    cont_cols = []

    means = {}

    stds = {}

    

    def __init__(self, cont_cols):

        self.cont_cols = cont_cols

        

    def apply_train(self, df):

        for col in self.cont_cols:

            self.means[col] = df[col].mean()

            self.stds[col] = df[col].std()

        self.apply_test(df)

        

    def apply_test(self, df):

        for col in self.cont_cols:

            df[col] = (df[col] - self.means[col])/(1e-7 + self.stds[col])

            

    def denorm(self, series):

        denormed = series * (1e-7 + self.stds[series.name]) + self.means[series.name]

        return denormed





"""

Convert categorical values into code, so that they can be used in Embeddings. 

"""

class Categorify():

    cat_cols = []

    categories = {}

    

    def __init__(self, cat_cols):

        self.cat_cols = cat_cols

        

    def apply_train(self, df):

        for col in self.cat_cols:

            df[col] = df[col].astype('category')

            self.categories[col] = df[col].cat.categories

            df[col] = df[col].cat.codes

    

    def apply_test(self, df):

        for col in self.cat_cols:

            df[col] = pd.Categorical(df[col],categories=self.categories[col])

            df[col] = df[col].cat.codes

            #if code is -1 (data is not present in training set, the set it  different category.Embedding size is nunique + 1)

            df[col].replace(-1,len(self.categories[col]),inplace=True)
fillMissing = FillMissing(cont_cols,cat_cols)

normalize_x = Normalize(cont_cols)

normalize_y = Normalize([target])

categorify = Categorify(cat_cols)



#preprocess training

fillMissing.apply_train(train)

normalize_x.apply_train(train)

categorify.apply_train(train)

normalize_y.apply_train(train)



#preprocess test

fillMissing.apply_test(test)

normalize_x.apply_test(test)

categorify.apply_test(test)
from collections import OrderedDict

emb_sizes = []

for col in cat_cols:

    ni = train[col].nunique() + 1

    nd = int(min(np.ceil(ni/2),50))

    emb_sizes.append((ni,nd))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



def get_ds(pdf, cat_cols, cont_cols, target,device,bs=1000):

    cat_tensor = torch.from_numpy(pdf[cat_cols].astype(np.int64).values).to(torch.long).to(device)

    cont_tensor = torch.from_numpy(pdf[cont_cols].values).to(torch.float32).to(device)

    if target:

        target_tensor = torch.from_numpy(pdf[target].values.astype(np.float32)).to(torch.float32).to(device)

    else:

        target_tensor = torch.zeros(pdf.shape[0]).to(device)  

        

    ds = TensorDataset(cat_tensor, cont_tensor, target_tensor)

    return ds



train_ds = get_ds(train,cat_cols,cont_cols, target,device=device)

test_ds = get_ds(test, cat_cols, cont_cols, target=None,device=device) 



#create val set

val_split = 0.15

val_size = int(train.shape[0] * val_split)

train_size = train.shape[0] - val_size

train_ds, val_ds = random_split(train_ds,(train_size,val_size))

train_dl = DataLoader(train_ds, 2000,shuffle=True)

val_dl = DataLoader(val_ds, 5000,shuffle=False)

print('train_size:val_size - {0}:{1}'.format(train_size,val_size))

from operator import add



class HousingModel(nn.Module):

    def __init__(self,cont_cols, emb_sizes,fc_sizes):

        super(HousingModel, self).__init__()

        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=emb_size[0],embedding_dim=emb_size[1]) for emb_size in emb_sizes])

        hidden_1 = len(cont_cols) + sum([e[1] for e in emb_sizes])

        hidden_sizes = [hidden_1] + fc_sizes

        self.fc_layers = nn.ModuleList([nn.Linear(hidden_sizes[o],hidden_sizes[o+1]) for o in range(len(fc_sizes))])

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_sizes[o]).to(device) for o in range(len(fc_sizes))])

        self.relu = nn.ReLU()

    

    def forward(self, xb_cat, xb_cont):

        if len(self.embeds) > 0:

            x_embs = [e(xb_cat[:,i]) for i, e in enumerate(self.embeds)]

            x_embs = torch.cat(x_embs, 1)

        x = torch.cat([x_embs, xb_cont],1)

        for idx, fc_layer in enumerate(self.fc_layers):

            x = self.fc_layers[idx](self.relu(self.bn_layers[idx](x)))

        return x.squeeze()



model = HousingModel(cont_cols,emb_sizes,fc_sizes=[64,1])

xb_cat, xb_cont, y = next(iter(train_dl))

print('xb_cat.shape: ', xb_cat.shape)

print('xb_cont.shape: ', xb_cont.shape)

yb_pred = model(xb_cat, xb_cont)

yb_pred.shape

xb_cont
from ignite.engine import Engine, Events

from ignite.metrics import Loss

from ignite.handlers import ModelCheckpoint, EarlyStopping

from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from ignite._utils import convert_tensor
def prep_batch(batch, device=None,non_blocking=False):

    x_cat, x_cont, y = batch

    return (convert_tensor(x_cat, device=device, non_blocking=non_blocking),

        convert_tensor(x_cont, device=device, non_blocking=non_blocking),

        convert_tensor(y, device=device, non_blocking=non_blocking))



def create_trainer(model, optimizer, loss_func,device,prepare_batch=prep_batch):

    if device:

        model.to(device)

    

    def _update_model(engine, batch):

        model.train()

        optimizer.zero_grad()

        x_cat, x_cont, y = prepare_batch(batch, device=device)

        y_pred = model(x_cat, x_cont)

        loss = loss_func(y_pred, y)

        loss.backward()

        optimizer.step()

        return loss.item()

    return Engine(_update_model)



def create_evaluator(model,metrics={},device=None, prepare_batch=prep_batch):

    if device:

        model.to(device)

        

    def _evaluate(engine, batch):

        model.eval()

        

        with torch.no_grad():

            x_cat, x_cont, y = prepare_batch(batch, device=device)

            y_pred = model(x_cat, x_cont)

            return y_pred, y

        

    engine = Engine(_evaluate)

    for name, metric in metrics.items():

        metric.attach(engine, name)

    return engine



def eval_submit(model, test_ds,test_df):

    #evaluate

    model.eval()

    y_preds = model(test_ds.tensors[0],test_ds.tensors[1])

    preds = normalize_y.denorm(pd.Series(y_preds.detach().numpy(),name=target))



    #create submission

    my_submission = pd.DataFrame({'Id': test_df.Id, target: preds})

    my_submission['Id'] = my_submission['Id'].astype(np.int32)

    my_submission.to_csv('submission.csv', index=False)





class PlotLosses(object):

    def __init__(self,evaluator,train_dl,val_dl):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

        self.evaluator = evaluator

        

    def __call__(self, engine):

        #get training loss

        evaluator.run(train_dl)

        train_metrics = evaluator.state.metrics

        self.losses.append(train_metrics['loss'])

        

        #get validation loss

        evaluator.run(val_dl)

        val_metrics = evaluator.state.metrics

        self.val_losses.append(val_metrics['loss'])

        

        self.x.append(self.i)

        self.i += 1

        #if (self.i)%10 == 0:

        #    print(engine.state.epoch, ' training loss: ',train_metrics['loss'],'\t val loss: ',val_metrics['loss'] )

            

        #plot

        clear_output(wait=True)

        plt.plot(self.x,self.losses,label='train_loss')

        plt.plot(self.x,self.val_losses,label='val_loss')

        plt.legend()

        plt.show()
model = HousingModel(cont_cols,emb_sizes,fc_sizes=[64,8,1])

optimizer = optim.Adam(model.parameters())

loss_func = nn.MSELoss(size_average = False) 



trainer = create_trainer(model, optimizer,loss_func,device,prepare_batch=prep_batch)

evaluator = create_evaluator(model,metrics={'loss':Loss(loss_func)})



#event handler to plot losses

plot_losses = PlotLosses(evaluator,train_dl,val_dl)

trainer.add_event_handler(Events.EPOCH_COMPLETED,plot_losses)



#start training

trainer.run(train_dl, max_epochs=100)
eval_submit(model, test_ds,test)

#this gave a score of 27117.28 and rank 5679
model = HousingModel(cont_cols,emb_sizes,fc_sizes=[64,8,1])

optimizer = optim.Adam(model.parameters())

loss_func = nn.MSELoss(size_average = False) 



trainer = create_trainer(model, optimizer,loss_func,device,prepare_batch=prep_batch)

evaluator = create_evaluator(model,metrics={'loss':Loss(loss_func)})



#event handler to plot losses

plot_losses = PlotLosses(evaluator,train_dl,val_dl)

trainer.add_event_handler(Events.EPOCH_COMPLETED,plot_losses)





#--> cosine annealing for lr

lr_scheduler = CosineAnnealingScheduler(optimizer,'lr',0.01,0.001,len(train_dl))

trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

trainer.run(train_dl, max_epochs=70)
eval_submit(model, test_ds,test)


model = HousingModel(cont_cols,emb_sizes,fc_sizes=[64,8,1])

optimizer = optim.Adam(model.parameters())

loss_func = nn.MSELoss(size_average = False) 



trainer = create_trainer(model, optimizer,loss_func,device,prepare_batch=prep_batch)

evaluator = create_evaluator(model,metrics={'loss':Loss(loss_func)})



#event handler to plot losses

# plot_losses = PlotLosses(evaluator,train_dl,val_dl)

# trainer.add_event_handler(Events.EPOCH_COMPLETED,plot_losses)



#--> cosine annealing for lr

lr_scheduler = CosineAnnealingScheduler(optimizer,'lr',0.01,0.001,len(train_dl))

trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)



train_ds = get_ds(train,cat_cols,cont_cols, target,device=device)

train_dl = DataLoader(train_ds,2000,shuffle=True)



trainer.run(train_dl, max_epochs=70)