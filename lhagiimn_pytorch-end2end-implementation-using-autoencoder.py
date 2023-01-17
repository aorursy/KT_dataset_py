import pandas as pd

import numpy as np

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pylab as plt

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch

import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau

import os

from tqdm import tqdm

import torch.nn.functional as F

import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import log_loss

# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import pickle



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess(df):

    df = df.copy()

    df['cp_type_trt'] = np.where(df['cp_type'].values == 'trt_cp', 1, 0)

    df['cp_type_ctl'] = np.where(df['cp_type'].values == 'trt_cp', 0, 1)

    df['cp_dose_D1'] = np.where(df['cp_dose'].values == 'D1', 1, 0)

    df['cp_dose_D2'] = np.where(df['cp_dose'].values == 'D1', 0, 1)

    df['cp_time_24'] = np.where(df['cp_time'].values == 24, 1, 0)

    df['cp_time_48'] = np.where(df['cp_time'].values == 48, 1, 0)

    df['cp_time_72'] = np.where(df['cp_time'].values == 72, 1, 0)

    return df



def make_X(dt, dense_cols, cat_feats):

    X = {"dense": dt[dense_cols].to_numpy()}

    for i, v in enumerate(cat_feats):

        X[v] = dt[[v]].to_numpy()

    return X





def get_data(ROOT = '../input/lish-moa'):



    cat_feat = ['cp_dose', 'cp_time']



    train = pd.read_csv(f"{ROOT}/train_features.csv")

    test = pd.read_csv(f"{ROOT}/test_features.csv")



    train['where'] = 'train'

    test['where'] = 'test'



    data = pd.concat([train, test], axis=0)



    # for var in data.iloc[:,4:-1].columns:

    #     data[var] = (data[var].values-data[var].mean())/data[var].std()



    label = pd.read_csv(f"{ROOT}/train_targets_scored.csv")

    label_test = pd.read_csv(f"{ROOT}/sample_submission.csv")



    data = preprocess(data)



    uniques = []

    for i, v in enumerate(cat_feat):

        data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]])

        uniques.append(len(data[v].unique()))



    FE = list(data)

    FE.remove('where')

    FE.remove('sig_id')

    FE.remove('cp_type_ctl')

    FE.remove('cp_type_trt')

    FE.remove('cp_type')

    for cat in cat_feat:

        FE.remove(cat)





    train = data.loc[data['where']=='train']

    test = data.loc[data['where'] == 'test']



    del data



    train = train.drop(['where'], axis=1)

    test = test.drop(['where'], axis=1)



    train = train.set_index('sig_id')

    test = test.set_index('sig_id')

    label = label.set_index('sig_id')

    label_test = label_test.set_index('sig_id')



    label = label.loc[train.index]

    label_test = label_test.loc[test.index]



    train = pd.concat([train, label], axis=1)

    test = pd.concat([test, label_test], axis=1)



    train['total'] = np.where(np.sum(train[list(label)].values, axis=1)>0, 0, 1)



    return train, test, FE, cat_feat, list(label), uniques
class block(nn.Module):

    def __init__(self, input_dim, keep_prob, hidden_dim):

        super(block, self).__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.dropout = nn.Dropout(keep_prob)

        self.dense = nn.Linear(input_dim, hidden_dim)



    def forward(self, x):

        x = self.batch_norm(x)

        x = self.dropout(x)

        x = self.dense(x)



        return x



class autoencoder(nn.Module):

    def __init__(self, input_dim):

        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 256),

            nn.ReLU(True),

            nn.BatchNorm1d(256),

            nn.Linear(256, 128),

            nn.ReLU(True), nn.BatchNorm1d(128), nn.Linear(128, 64))

        self.decoder = nn.Sequential(

            nn.Linear(64, 128),

            nn.ReLU(True),

            nn.BatchNorm1d(128),

            nn.Linear(128, 256),

            nn.ReLU(True),

            nn.BatchNorm1d(256),

            nn.Linear(256, input_dim))



    def forward(self, x):

        z = self.encoder(x)

        x = self.decoder(z)

        return x, z

        

class MoaModel_encoder(nn.Module):

    def __init__(self, hidden_dim, emb_dims, n_cont):

        super(MoaModel_encoder, self).__init__()



        self.auto_encoder = autoencoder(n_cont)



        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        n_embs = sum([y for x, y in emb_dims])

        num_columns = n_embs + n_cont + 64



        self.block1 = block(num_columns, 0.25, hidden_dim)

        self.block2 = block(hidden_dim, 0.5, int(hidden_dim/2))

        self.block3 = block(int(hidden_dim / 2), 0.25, 206)



    def encode_and_combine_data(self, cat_data):

        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]

        xcat = torch.cat(xcat, 1)

        return xcat



    def forward(self, cont_data, cat_data):



        cont_data = cont_data.to(device)

        cat_data = cat_data.to(device)



        re_cont_data, low_dim = self.auto_encoder(cont_data)



        error = torch.abs(re_cont_data-cont_data)

        cat_data = self.encode_and_combine_data(cat_data)

        x = torch.cat([low_dim, cat_data, error], dim=1)





        x = F.relu(self.block1(x))

        x = F.relu(self.block2(x))

        x = F.sigmoid(self.block3(x))



        return x, re_cont_data
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

## Early stopping algorithm

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
## Model training

def model_training(model, train_loader, val_loader, loss_function,

                   epochs,

                   lr=0.001, patience=10,

                   model_path='model.pth'):







    if os.path.isfile(model_path):



        # load the last checkpoint with the best model

        model = torch.load(model_path)



        return model



    else:



        # Loss and optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,

                                      factor=0.5, verbose=True)



        criteria = loss_function



        train_losses = []

        val_losses = []

        early_stopping = EarlyStopping(patience=patience, verbose=True)



        for epoch in tqdm(range(epochs)):



            train_loss, val_loss = 0, 0



            # Training phase

            model.train()

            bar = tqdm(train_loader)



            for i, (X_cont, X_cat, y) in enumerate(bar):

                preds, cont_data_x = model(X_cont, X_cat)



                loss = criteria(preds.flatten().unsqueeze(1), y.to(device).flatten().unsqueeze(1)) + 0.5*F.mse_loss(X_cont.to(device), cont_data_x)



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                with torch.no_grad():

                    train_loss += loss.item() / (len(train_loader))

                    bar.set_description(f"{loss.item():.3f}")



            # Validation phase

            val_preds = []

            true_y = []

            model.eval()

            with torch.no_grad():

                for i, (X_cont, X_cat, y) in enumerate(val_loader):

                    preds, cont_data_x = model(X_cont, X_cat)



                    val_preds.append(preds)

                    true_y.append(y)



                    loss = criteria(preds.flatten().unsqueeze(1),y.to(device).flatten().unsqueeze(1))  # + F.mse_loss(X_cont.to(device), cont_data_x)

                    val_loss += loss.item() / (len(val_loader))



                score = F.binary_cross_entropy(torch.cat(val_preds, dim=0), torch.cat(true_y, dim=0).to(device))



            print(f"[{'Val'}] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val score: {score:.4f}")



            early_stopping(score, model, path=model_path)



            if early_stopping.early_stop:

                print("Early stopping")

                break



            train_losses.append(train_loss)

            val_losses.append(val_loss)

            scheduler.step(score)



        model = torch.load(model_path)

        return model





def fully_train(model, train_data, cont_features,

                cat_features, labels, kf,

                loss_function, hidden_dim, emb_dims,

                epochs, lr, patience,

                kfold=5, model_path_temp='model'):



    models = []

    val_loaders = []

    val_set = []



    for i, (train_index, test_index) in enumerate(kf.split(train_data, train_data[labels])):

        print('[Fold %d/%d]' % (i + 1, kfold))



        model_path = "%s_%s.pth" % (model_path_temp, i)



        X_train, valX = train_data.iloc[train_index], train_data.iloc[test_index]

        X_train = X_train.loc[X_train['cp_type_ctl'] != 1, :]

        X_val = valX.loc[valX['cp_type_ctl'] != 1, :]

        y_train, y_valid = X_train[labels].values, X_val[labels].values



        X_train = make_X(X_train.reset_index(), cont_features, cat_features)

        X_valid = make_X(X_val.reset_index(), cont_features, cat_features)





        train_loader = Loader(X_train, y_train, cat_cols=cat_features, batch_size=128, shuffle=True)

        val_loader = Loader(X_valid, y_valid, cat_cols=cat_features, batch_size=256, shuffle=False)



        model_temp = model(hidden_dim, emb_dims, len(cont_features)).to(device)

        print(model_temp)

        exit()

        model_temp = model_training(model_temp, train_loader, val_loader, loss_function=loss_function,

                               epochs=epochs,

                               lr=lr, patience=patience,

                               model_path=model_path)



        models.append(model_temp)

        val_loaders.append(val_loader)

        val_set.append(valX)



    return models, val_loaders, val_set
train, test, FE, cat_feat, labels, uniques = get_data()



dims = [2, 8]

emb_dims = [(x, y) for x, y in zip(uniques, dims)]

n_cont = len(FE)



Nets = [MoaModel_encoder]

Net_names = ['MoaModel_encoder']



#Hyperparameters

hidden_dim = 512

kfold = 5

skf = KFold(n_splits=kfold, shuffle=True, random_state=45)

#skf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state=128)



all_models = []

val_loaders = []

val_sets =[]



for model, name in zip(Nets, Net_names):

    models, val_loader, val_set = fully_train(model=model, train_data = train,

                                      cont_features = FE, cat_features = cat_feat,

                                      labels=labels, kf=skf,  loss_function = nn.BCELoss(),

                                      hidden_dim=hidden_dim, emb_dims=emb_dims,

                                      epochs=1000, lr=0.01, patience=10,

                                      kfold=kfold, model_path_temp=name)



    all_models.append(models)

    val_loaders.append(val_loader)

    val_sets.append(val_set)





scores = []



y_val_avg = []

for i in range(len(Nets)):

    for kf in range(kfold):

        temp_pred = []

        temp_y = []

        with torch.no_grad():

            for X_cont, X_cat, y in val_loaders[i][kf]:

                preds, _ = all_models[i][kf](X_cont, X_cat)

                temp_pred.append(preds)

                temp_y.append(y)



        y_pred = torch.cat(temp_pred, dim=0).detach().cpu().numpy()

        #y_true = torch.cat(temp_y, dim=0).detach().cpu().numpy()



        val_temp_set = val_sets[i][kf]

        y_true = val_temp_set[labels].values



        val_temp_set.loc[val_temp_set['cp_type_ctl'] != 1, labels] = y_pred

        y_pred = val_temp_set[labels].values



        # plt.plot(np.sort(y_pred.flatten()))

        # plt.show()





        score = 0

        for k in range(y_true.shape[1]):

            score_ = log_loss(y_true[:, k], y_pred[:, k].astype(float), labels=[0,1])

            score += score_ / y_true.shape[1]



        print('Fold %s:' % kf, score)

        scores.append(score)



print('#'*150)

print('CV average:', np.mean(scores))

print('CV std:', np.std(scores))

print('#'*150)



X_test = make_X(test, FE, cat_feat)

test_loader = Loader(X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False)



full_test = np.zeros([test.shape[0], 206, len(Nets)*kfold])

for i in range(len(Nets)):

    for kf in range(kfold):

        temp_pred = []

        temp_y = []

        with torch.no_grad():

            for X_cont, X_cat, y in test_loader:

                preds, _ = all_models[i][kf](X_cont, X_cat)

                temp_pred.append(preds)

                temp_y.append(y)



        full_test[:, :, i*kfold+kf] = torch.cat(temp_pred, dim=0).detach().cpu().numpy()





#test = test[labels]

print(full_test.shape)

print(np.mean(full_test, axis=2).shape)

test[labels] = np.mean(full_test, axis=2)

test.loc[test['cp_type_ctl']==1, labels]=0

test[labels].to_csv('submission.csv')