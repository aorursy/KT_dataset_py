# import libraries



import numpy as np

import pandas as pd

import pydicom

import os

import random

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold,  GroupKFold

from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

import torch.nn as nn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder



import warnings



warnings.simplefilter('ignore')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantiles = [0.2, 0.5, 0.8]
# dataframe for loader

def make_X(dt, dense_cols, cat_feats):

    X = {"dense": dt[dense_cols].to_numpy()}

    for i, v in enumerate(cat_feats):

        X[v] = dt[[v]].to_numpy()

    return X



# loader for embedding layers

class Loader:



    def __init__(self, X, y, shuffle=True, batch_size=64, cat_cols=[]):



        self.X_cont = X["dense"]

        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)

        self.y = y



        self.shuffle = shuffle

        self.batch_size = batch_size

        self.n_conts = self.X_cont.shape[1]

        self.len = self.X_cont.shape[0]

        n_batches, remainder = divmod(self.len, self.batch_size)



        if remainder > 0:

            n_batches += 1

        self.n_batches = n_batches

        self.remainder = remainder  # for debugging



        self.idxes = np.array([i for i in range(self.len)])



    def __iter__(self):

        self.i = 0

        if self.shuffle:

            ridxes = self.idxes

            np.random.shuffle(ridxes)

            self.X_cat = self.X_cat[[ridxes]]

            self.X_cont = self.X_cont[[ridxes]]

            if self.y is not None:

                self.y = self.y[[ridxes]]



        return self



    def __next__(self):

        if self.i >= self.len:

            raise StopIteration



        if self.y is not None:

            y = torch.FloatTensor(self.y[self.i:self.i + self.batch_size].astype(np.float32))



        else:

            y = None



        xcont = torch.FloatTensor(self.X_cont[self.i:self.i + self.batch_size])

        xcat = torch.LongTensor(self.X_cat[self.i:self.i + self.batch_size])



        batch = (xcont, xcat, y)

        self.i += self.batch_size

        return batch



    def __len__(self):

        return self.n_batches


class model_nn(nn.Module):



    def __init__(self, hidden_dim, output_dim, emb_dims, n_cont):

        super().__init__()



        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        n_embs = sum([y for x, y in emb_dims])



        self.n_embs = n_embs  # + t_embs

        self.n_cont = n_cont



        inp_dim = n_embs + n_cont

        self.inp_dim = inp_dim

        

        self.fs0 = nn.Linear(inp_dim, hidden_dim)

        self.relufs0 = nn.ELU()

        self.fs1 = nn.Linear(hidden_dim, inp_dim)

        self.fs2 = nn.Sigmoid()

        

        self.fc0 = nn.Linear(inp_dim, hidden_dim)

        self.relu0 = nn.ELU()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.relu1 = nn.ELU()



        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.fc4 = nn.Linear(hidden_dim, output_dim)



    def encode_and_combine_data(self, cat_data):

        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]

        xcat = torch.cat(xcat, 1)

        return xcat



    def forward(self, cont_data, cat_data):

        cont_data = cont_data.to(device)

        cat_data = cat_data.to(device)



        cat_data = self.encode_and_combine_data(cat_data)



        x = torch.cat([cont_data, cat_data], dim=1)

        

        w = self.fs0(x)

        w = self.relufs0(w)

        w = self.fs1(w)

        w = self.fs2(w)

        

        wx = w*x + x

        

        hz = self.fc0(wx)

        hz = self.relu0(hz)

        hz = self.fc1(hz)

        hz = self.relu1(hz)



        out1 = self.fc2(hz)

        out2 = self.fc3(hz)

        out3 = self.fc4(hz)

        

        return torch.cat([out1, out2, out3], dim=1)
#loss function

def quantile_loss(preds, target, quantiles):

    assert not target.requires_grad

    assert preds.size(0) == target.size(0)

    losses = []

    for i, q in enumerate(quantiles):

        errors = target - preds[:, i]

        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss



#competation metric for numpy array

def metric(outputs, target):



    confidence = np.abs(outputs[:, 2] - outputs[:, 0])

    clip = np.where(confidence > 70, confidence, 70)

    delta = np.abs(outputs[:, 1] - target)

    delta = np.where(delta > 1000, 1000, delta)



    metrics = (delta*np.sqrt(2)/clip) + np.log(clip*np.sqrt(2))



    return np.mean(metrics)
class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):



        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta



    def __call__(self, val_loss, model, path):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score - self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

            self.counter = 0



    def save_checkpoint(self, val_loss, model, path):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model, path)

        self.val_loss_min = val_loss



def model_training(model, train_loader, val_loader, epochs,

                   batch_size=64, lr=0.001, patience=10,

                   model_path='model.pth'):

    

    if os.path.isfile(model_path):



        # load the last checkpoint with the best model

        model = torch.load(model_path)



        return model



    else:



        # Loss and optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,

                                      factor=0.4, verbose=True)



        train_losses = []

        val_losses = []

        early_stopping = EarlyStopping(patience=patience, verbose=True)



        for epoch in tqdm(range(epochs)):

            train_loss, val_loss = 0, 0

            # Training phase

            model.train()

            bar = tqdm(train_loader)



            for i, (X_cont, X_cat, y) in enumerate(bar):

                preds = model(X_cont, X_cat)

                loss = quantile_loss(preds, y.to(device), quantiles)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                with torch.no_grad():

                    train_loss += loss.item() / len(train_loader)

                    # bar.set_description(f"{loss.item():.3f}")



            # Validation phase

            val_preds = []

            true_y = []

            model.eval()

            with torch.no_grad():

                for phase in ["valid"]:

                    if phase == "train":

                        loader = train_loader

                    else:

                        loader = val_loader



                    for i, (X_cont, X_cat, y) in enumerate(loader):

                        preds = model(X_cont, X_cat)



                        val_preds.append(preds)

                        true_y.append(y)



                        loss = quantile_loss(preds, y.to(device), quantiles)

                        val_loss += loss.item() / len(loader)



                val_preds = torch.cat(val_preds, dim=0).detach().cpu().numpy()

                true_y = torch.cat(true_y, dim=0).detach().cpu().numpy()

                score = metric(val_preds, true_y)



            print(f"[{phase}] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val score: {score:.4f}")



            early_stopping(score, model, path=model_path)



            if early_stopping.early_stop:

                print("Early stopping")

                break



            train_losses.append(train_loss)

            val_losses.append(val_loss)

            scheduler.step(val_loss)



        model = torch.load(model_path)



        return model
ROOT = "../input/osic-pulmonary-fibrosis-progression"

Model_Root = 'models'  #this root for the prediction phase 


tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")



tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])



data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



base = data.loc[data.Weeks == data.min_week]

base = base[['Patient', 'FVC', 'Percent']].copy()

base.columns = ['Patient','min_FVC', 'min_percent']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)



data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base



COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)



data['age_week'] = data['Age'].values + data['base_week'].values/53

        

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['min_percent_norm'] = (data['min_percent'] - data['min_percent'].min() ) / ( data['min_percent'].max() - data['min_percent'].min() )

data['age_week_norm'] = (data['age_week'] - data['age_week'].min() ) / ( data['age_week'].max() - data['age_week'].min() )



cat_feat = ['Sex', 'SmokingStatus']



uniques = []

for i, v in enumerate(cat_feat):

    data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]])

    uniques.append(len(data[v].unique()))



tr = data.loc[data.WHERE == 'train']

chunk = data.loc[data.WHERE == 'val']

sub = data.loc[data.WHERE == 'test']



FE += ['age', 'week', 'BASE', 'min_percent_norm', 'age_week_norm']



print(list(tr))

print(list(sub))

print(FE)

print(cat_feat)

print(uniques)



dims = [1, 2]

emb_dims = [(x, y) for x, y in zip(uniques, dims)]

n_cont = len(FE)
# model training and submission print



hidden_dim = 256

out_dim = 1



kfold = 6

groups = np.asarray(tr['Patient'].values)

skf = GroupKFold(n_splits=kfold)



avg_preds = np.zeros((len(sub), len(quantiles)))

models = []



for i, (train_index, test_index) in enumerate(skf.split(tr, tr['FVC'].values, groups=groups)):

    print('[Fold %d/%d]' % (i + 1, kfold))



    model_path = f"{Model_Root}/nn_model_%s.pth" % i



    if os.path.isfile(model_path):



        final_model = torch.load(model_path)



        X_test = make_X(sub, FE, cat_feat)

        test_loader = Loader(X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False)



        preds = []

        # model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)

        # model.load_state_dict(torch.load(model_path))





        for i, (X_cont, X_cat, y) in enumerate(tqdm(test_loader)):

            print(X_cont.shape, X_cat.shape)

            out = final_model(X_cont, X_cat)

            preds.append(out)



        preds = torch.cat(preds, dim=0).detach().cpu().numpy()

        avg_preds += preds



    else:



        X_train, X_valid = tr.iloc[train_index], tr.iloc[test_index]

        y_train, y_valid = tr.iloc[train_index]['FVC'].values, tr.iloc[test_index]['FVC'].values



        X_train = make_X(X_train.reset_index(), FE, cat_feat)

        X_valid = make_X(X_valid.reset_index(), FE, cat_feat)



        train_loader = Loader(X_train, y_train, cat_cols=cat_feat, batch_size=16, shuffle=True)

        val_loader = Loader(X_valid, y_valid, cat_cols=cat_feat, batch_size=64, shuffle=True)



        model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)



        final_model = model_training(model, train_loader, val_loader, epochs=1000,

                                     batch_size=64, lr=0.01, patience=20,

                                     model_path='nn_model_%s.pth' % i)



        models.append(final_model)



for model in models:



    X_test = make_X(sub, FE, cat_feat)

    test_loader = Loader(X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False)



    preds = []

    # model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)

    # model.load_state_dict(torch.load('nn_model_%s.pth' % i))

    with torch.no_grad():

        for i, (X_cont, X_cat, y) in enumerate(tqdm(test_loader)):

            out = model(X_cont, X_cat)

            preds.append(out)



    preds = torch.cat(preds, dim=0).detach().cpu().numpy()

    avg_preds += preds



print(avg_preds)

avg_preds = avg_preds/kfold

print(avg_preds)



sub['Patient_Week'] = sub['Patient_Week'].values

sub['FVC'] = avg_preds[:, 1]

sub['Confidence'] = avg_preds[:, 2]-avg_preds[:, 0]



sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)