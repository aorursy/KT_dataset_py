import copy

import os

import time

import traceback

from contextlib import contextmanager

from typing import List



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.nn as nn



!pip install nyaggle

from nyaggle.validation import StratifiedGroupKFold

from sklearn.cluster import AgglomerativeClustering

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
TOKEN2INT = {x: i for i, x in enumerate('().ACGUBEHIMSX')}

PRED_COLS_SCORED = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

PRED_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']



DATA_DIR = "../input/stanford-covid-vaccine/"

REPLACE_DATA_PATH = "../input/eternafold/eternafold_mfe.csv"

PRIMARY_BPPS_DIR = "../input/eternafold/bpps/"

SECONDARY_BPPS_DIR = "../input/bpps-by-viennat70/"

NFOLDS = 7

BATCH_SIZE = 64

TRAIN_EPOCHS = 140
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )



    def forward(self, x):

        b, c, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)





class Conv(nn.Module):

    def __init__(self, d_in, d_out, kernel_size, dropout=0.1):

        super().__init__()

        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)

        self.bn = nn.BatchNorm1d(d_out)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)



    def forward(self, src):

        return self.dropout(self.relu(self.bn(self.conv(src))))





class ResidualGraphAttention(nn.Module):

    def __init__(self, d_model, kernel_size, dropout):

        super().__init__()

        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)

        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)

        self.relu = nn.ReLU()



    def forward(self, src, attn):

        h = self.conv2(self.conv1(torch.bmm(src, attn)))

        return self.relu(src + h)

    



class SEResidual(nn.Module):

    def __init__(self, d_model, kernel_size, dropout):

        super().__init__()

        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)

        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)

        self.relu = nn.ReLU()

        self.se = SELayer(d_model)



    def forward(self, src):

        h = self.conv2(self.conv1(src))

        return self.se(self.relu(src + h))





class FusedEmbedding(nn.Module):

    def __init__(self, n_emb):

        super().__init__()

        self.emb = nn.Embedding(len(TOKEN2INT), n_emb)

        self.n_emb = n_emb



    def forward(self, src, se):

        # src: [batch, seq, feature]

        # se: [batch, seq]

        embed = self.emb(src)

        embed = embed.reshape((-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        embed = torch.cat((embed, se), 2)



        return embed



    @property

    def d_out(self):

        d_emb = 3 * self.n_emb

        d_feat = 2 * 5  # max, sum, 2nd, 3rd, nb_count

        return d_emb + d_feat

    



class ConvModel(nn.Module):

    def __init__(self, d_emb=50, d_model=256, dropout=0.6, dropout_res=0.4, dropout_emb=0.0,

                 kernel_size_conv=7, kernel_size_gc=7):

        super().__init__()



        self.embedding = FusedEmbedding(d_emb)

        self.dropout = nn.Dropout(dropout_emb)

        self.conv = Conv(self.embedding.d_out, d_model, kernel_size=3, dropout=dropout)



        self.block1 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)

        self.block2 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)

        self.block3 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)

        self.block4 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)

        self.block5 = SEResidual(d_model, kernel_size=kernel_size_conv, dropout=dropout_res)



        self.attn1 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)

        self.attn2 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)

        self.attn3 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)

        self.attn4 = ResidualGraphAttention(d_model, kernel_size=kernel_size_gc, dropout=dropout_res)



        self.linear = nn.Linear(d_model, len(PRED_COLS))



    def forward(self, 

                src: torch.Tensor, 

                features: torch.Tensor, 

                bpps: torch.Tensor, 

                adj: torch.Tensor):

        # src: [batch, seq, 3]

        # features: [batch, seq, 10]

        # bpps: [batch, seq, seq, 2]

        # adj: [batch, seq, seq]

        

        x = self.dropout(self.embedding(src, features))

        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]

        

        x = self.conv(x)

        x = self.block1(x)

        x = self.attn1(x, adj)

        x = self.block2(x)

        x = self.attn2(x, adj)

        x = self.block3(x)

        x = self.attn3(x, bpps[:, :, :, 0])

        x = self.attn4(x, bpps[:, :, :, 1])

        x = self.block4(x)

        x = self.block5(x)



        x = x.permute([0, 2, 1])  # [batch, seq, features]

        out = self.linear(x)



        out = torch.clamp(out, -0.5, 1e8)



        return out

    





class WRMSELoss(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, yhat, y, sample_weight=None):

        l = (yhat - y) ** 2



        if sample_weight is not None:

            l = l * sample_weight.unsqueeze(dim=1)



        return torch.sqrt(torch.mean(l))





class ColWiseLoss(nn.Module):

    def __init__(self, base_loss):

        super().__init__()

        self.base_loss = base_loss

        self.len_scored = 68



    def forward(self, yhat, y, column_weight=None, sample_weight=None):

        score = 0

        for i in range(len(PRED_COLS)):

            s = self.base_loss(

                yhat[:, :self.len_scored, i], 

                y[:, :self.len_scored, i], 

                sample_weight

            ) / len(PRED_COLS)

            

            if column_weight is not None:

                s *= column_weight[i]

                

            score += s

        return score





class MCRMSELoss(ColWiseLoss):

    def __init__(self):

        super().__init__(WRMSELoss())
@contextmanager

def timer(name):

    s = time.time()

    yield

    print(f"{name}: {time.time() - s:.3f}sec")





def pandas_list_to_array(df: pd.DataFrame) -> np.ndarray:

    return np.transpose(

        np.array(

            df.values

                .tolist()

        ),

        (0, 2, 1)

    )





def preprocess_inputs(df: pd.DataFrame) -> np.ndarray:

    return pandas_list_to_array(

        df[['sequence', 'structure', 'predicted_loop_type']]

            .applymap(lambda seq: [TOKEN2INT[x] for x in seq])

    )





def build_adj_matrix(src_df: pd.DataFrame, normalize: bool = True) -> np.ndarray:

    n = len(src_df['structure'].iloc[0])

    mat = np.zeros((len(src_df), n, n))

    start_token_indices = []



    for r, structure in tqdm(enumerate(src_df['structure'])):

        for i, token in enumerate(structure):

            if token == "(":

                start_token_indices.append(i)

            elif token == ")":

                j = start_token_indices.pop()

                mat[r, i, j] = 1

                mat[r, j, i] = 1



    assert len(start_token_indices) == 0



    if normalize:

        mat = mat / (mat.sum(axis=2, keepdims=True) + 1e-8)



    return mat





def replace_data(train_df: pd.DataFrame, test_df: pd.DataFrame, replace_data_dir: str):

    print(f"using data from {replace_data_dir}")



    aux = pd.read_csv(replace_data_dir)

    del train_df['structure']

    del train_df['predicted_loop_type']

    del test_df['structure']

    del test_df['predicted_loop_type']

    train_df = pd.merge(train_df, aux, on='id', how='left')

    test_df = pd.merge(test_df, aux, on='id', how='left')

    assert len(train_df) == 2400

    assert len(test_df) == 3634

    assert train_df['structure'].isnull().sum() == 0

    assert train_df['predicted_loop_type'].isnull().sum() == 0

    assert test_df['structure'].isnull().sum() == 0

    assert test_df['predicted_loop_type'].isnull().sum() == 0

    return train_df, test_df





def load_bpps(df: pd.DataFrame, data_dir: str) -> np.ndarray:

    return np.array([np.load(f'{data_dir}bpps/{did}.npy') for did in df.id])





def make_bpps_features(bpps_list: List[np.ndarray]) -> np.ndarray:

    ar = []



    for b in bpps_list:

        ar.append(b.sum(axis=2))



        # max, 2ndmax, 3rdmax

        bpps_sorted = np.sort(b, axis=2)[:, :, ::-1]

        ar.append(bpps_sorted[:, :, 0])

        ar.append(bpps_sorted[:, :, 1])

        ar.append(bpps_sorted[:, :, 2])



        # number of nonzero

        bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data

        bpps_nb_std = 0.08914  # std of bpps_nb across all training data

        nb = (b > 0).sum(axis=2)

        nb = (nb - bpps_nb_mean) / bpps_nb_std

        ar.append(nb)



    return np.transpose(np.array(ar), (1, 2, 0))





def make_dataset(device, x: np.ndarray, y: np.ndarray,

                 bpps_primary: np.ndarray,

                 bpps_secondary: np.ndarray,

                 adj_matrix: np.ndarray,

                 prediction_mask: np.ndarray,

                 signal_to_noise=None):

    x = copy.deepcopy(x)

    if y is not None:

        y = copy.deepcopy(y)

    bpps_primary = copy.deepcopy(bpps_primary)

    bpps_secondary = copy.deepcopy(bpps_secondary)

    bpps = np.concatenate([

        bpps_primary[:, :, :, np.newaxis],

        bpps_secondary[:, :, :, np.newaxis]

    ], axis=-1)



    adj_matrix = copy.deepcopy(adj_matrix)

    prediction_mask = copy.deepcopy(prediction_mask)



    if y is not None:

        y = np.clip(y, -0.5, 10)

        mask = np.abs(y).max(axis=(1, 2)) < 10

    else:

        mask = [True] * len(x)



    tensors = [

        torch.LongTensor(x[mask]),

        torch.Tensor(make_bpps_features([bpps_primary[mask], bpps_secondary[mask]])),

        torch.Tensor(bpps[mask]),

        torch.Tensor(adj_matrix[mask]),

        torch.Tensor(prediction_mask[mask])

    ]



    if y is not None:

        tensors.append(torch.Tensor(y[mask]))

        

        sample_weight = np.clip(np.log(signal_to_noise[mask] + 1.1) / 2, 0, 100)

        tensors.append(torch.Tensor(sample_weight))



    return torch.utils.data.TensorDataset(*[t.to(device) for t in tensors])





def make_dataset_from_df(device, df: pd.DataFrame, bpps_dir: str, secondary_bpps_dir: str):

    assert df['seq_scored'].nunique() == 1



    inputs = preprocess_inputs(df)

    bpps = load_bpps(df, bpps_dir)

    adj = build_adj_matrix(df)

    secondary_bpps = load_bpps(df, secondary_bpps_dir)



    mask = np.zeros((len(df), len(df['sequence'].iloc[0]), len(PRED_COLS)))

    mask[:, :df['seq_scored'].iloc[0], :] = 1



    return make_dataset(device, inputs, None, bpps, secondary_bpps, adj, mask)





def dist(s1: str, s2: str) -> int:

    return sum([c1 != c2 for c1, c2 in zip(s1, s2)])





def get_distance_matrix(s: pd.Series) -> np.ndarray:

    mat = np.zeros((len(s), len(s)))



    for i in tqdm(range(len(s))):

        for j in range(i + 1, len(s)):

            mat[i, j] = mat[j, i] = dist(s[i], s[j])

    return mat





def batch_predict(model: nn.Module, loader: DataLoader) -> np.ndarray:

    y_preds = np.zeros((len(loader.dataset), loader.dataset[0][0].shape[0], len(PRED_COLS)))



    for i, (x_batch, x_se, x_bpps, x_adj, y_mask) in enumerate(loader):

        y_pred = model(x_batch, x_se, x_bpps, x_adj).detach() * y_mask

        y_preds[i * loader.batch_size:(i + 1) * loader.batch_size, :, :] = y_pred.cpu().numpy()

        

    return y_preds





def calc_loss(y_true: np.ndarray, y_pred: np.ndarray):

    err_w_valid = [1 if s in PRED_COLS_SCORED else 0 for s in PRED_COLS]

    raw = MCRMSELoss()(torch.Tensor(y_pred), torch.Tensor(y_true), err_w_valid).item()

    

    return raw * len(PRED_COLS) / len(PRED_COLS_SCORED)
def train_model(model, train_loader, valid_loader, y_valid,

                train_epochs, train_loss, verbose=True,

                model_path='model'):

    params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(f'number of params: {params}')



    err_w_train_1 = [1 if s in PRED_COLS_SCORED else 1.0 for s in PRED_COLS]

    err_w_train_2 = [1 if s in PRED_COLS_SCORED else 0.01 for s in PRED_COLS]



    criterion_train = train_loss

    optimizer = torch.optim.Adam(model.parameters())



    losses = []

    val_losses = []

    y_preds_best = None



    for epoch in range(train_epochs):

        start_time = time.time()



        model.train()

        avg_loss = 0.



        for x_batch, x_se, x_bpps, x_adj, y_mask, y_batch, sample_weight in tqdm(train_loader, disable=True):

            y_pred = model(x_batch, x_se, x_bpps, x_adj) * y_mask

            

            # use 5 columns for the first 30 epoch

            w = err_w_train_1 if epoch < 30 else err_w_train_2

            

            loss = criterion_train(y_pred, y_batch, w, sample_weight)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)



        model.eval()

        y_preds = batch_predict(model, valid_loader)

        mcloss = calc_loss(y_valid, y_preds)

        val_losses.append(mcloss)

        

        s = f"{epoch:03d}: trn:{avg_loss:.4f}, clean={mcloss:.4f}, {time.time() - start_time:.2f}s"



        losses.append(avg_loss)



        if np.min(val_losses) == mcloss:

            y_preds_best = y_preds

            torch.save(model.state_dict(), model_path)



        if (isinstance(verbose, bool) and verbose) or (verbose > 0 and (epoch % verbose == 0)):

            print(s)



    print(f'min val_loss: {np.min(val_losses):.4f} at {np.argmin(val_losses) + 1} epoch')



    # recover best weight

    model.load_state_dict(torch.load(model_path))



    if not verbose:

        return np.min(val_losses)



    fig, ax = plt.subplots(1, 3, figsize=(24, 8))



    ax[0].plot(np.arange(1, len(losses) + 1), losses)

    ax[1].plot(np.arange(1, len(val_losses) + 1), val_losses)



    for i, p in enumerate(PRED_COLS):

        ax[2].scatter(y_valid[:, :, i].flatten(), y_preds_best[:, :, i].flatten(), alpha=0.5)



    ax[0].legend(['train'])

    ax[1].legend(['valid(clean)'])

    ax[2].legend(PRED_COLS)

    ax[2].set_xlabel('y_true')

    ax[2].set_ylabel('y_predicted')

    ax[0].set_xlabel('epoch')

    ax[0].set_ylabel('loss')

    ax[1].set_xlabel('epoch')

    ax[1].set_ylabel('loss')

    ax[2].set_xlabel('y_true(clean)')

    ax[2].set_ylabel('y_predicted(clean)')

    plt.show()



    return np.min(val_losses)
with timer("load data"):

    train_df = pd.read_json(DATA_DIR + 'train.json', lines=True)

    test_df = pd.read_json(DATA_DIR + 'test.json', lines=True)

    sample_df = pd.read_csv(DATA_DIR + 'sample_submission.csv')



    train_df, test_df = replace_data(train_df, test_df, REPLACE_DATA_PATH)



with timer("clustering"):

    # use clustering based on edit distance

    seq_dist = get_distance_matrix(train_df['sequence'])

    clf = AgglomerativeClustering(n_clusters=None, 

                                  distance_threshold=20, 

                                  affinity='precomputed',

                                  linkage='average')

    group_index = clf.fit_predict(seq_dist)
with timer("preprocess"):

    public_df = test_df.query("seq_length != 130")

    private_df = test_df.query("seq_length == 130")



    x = preprocess_inputs(train_df)

    y = pandas_list_to_array(train_df[PRED_COLS])



    label_mask = np.ones_like(y)

    pad = np.zeros((y.shape[0], x.shape[1] - y.shape[1], y.shape[2]))

    y = np.concatenate((y, pad), axis=1)

    label_mask = np.concatenate((label_mask, pad), axis=1)



    assert x.shape[1] == y.shape[1]



    train_adj = build_adj_matrix(train_df)

    primary_bpps = load_bpps(train_df, PRIMARY_BPPS_DIR)    

    secondary_bpps = load_bpps(train_df, SECONDARY_BPPS_DIR)



    public_data = make_dataset_from_df(device, public_df, PRIMARY_BPPS_DIR, SECONDARY_BPPS_DIR)

    private_data = make_dataset_from_df(device, private_df, PRIMARY_BPPS_DIR, SECONDARY_BPPS_DIR)
kf = StratifiedGroupKFold(NFOLDS, random_state=42, shuffle=True)



pred_oof = np.zeros_like(y)

pred_public = np.zeros((len(public_data), len(public_df['sequence'].iloc[0]), len(PRED_COLS)))

pred_private = np.zeros((len(private_data), len(private_df['sequence'].iloc[0]), len(PRED_COLS)))



public_loader = DataLoader(public_data, batch_size=128, shuffle=False)

private_loader = DataLoader(private_data, batch_size=128, shuffle=False)



clean_idx = [i for i in range(len(train_df)) if train_df['SN_filter'].iloc[i]]

sn_mask = train_df['SN_filter'] == 1



criterion_train = MCRMSELoss()

model_path = "model_fold{}"



losses = []



for i, (train_index, valid_index) in enumerate(kf.split(x, train_df['SN_filter'], groups=group_index)):

    print(f'fold {i}')

    model = ConvModel().to(device)

    s = time.time()



    train_data = make_dataset(device, x[train_index], y[train_index],

                              primary_bpps[train_index], secondary_bpps[train_index],

                              train_adj[train_index],

                              label_mask[train_index],

                              signal_to_noise=train_df['signal_to_noise'][train_index].values)



    valid_index_c = [v for v in valid_index if v in clean_idx]

    valid_data_clean = make_dataset(device, x[valid_index_c], None,

                                    primary_bpps[valid_index_c],

                                    secondary_bpps[valid_index_c],

                                    train_adj[valid_index_c],

                                    label_mask[valid_index_c])

    valid_data_noisy = make_dataset(device, x[valid_index], None,

                                  primary_bpps[valid_index],

                                  secondary_bpps[valid_index],

                                  train_adj[valid_index],

                                  label_mask[valid_index])



    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_loader_clean = DataLoader(valid_data_clean, batch_size=128, shuffle=False)

    valid_loader_noisy = DataLoader(valid_data_noisy, batch_size=128, shuffle=False)



    loss = train_model(model, 

                       train_loader, 

                       valid_loader_clean, 

                       y[valid_index_c], 

                       TRAIN_EPOCHS, 

                       criterion_train,

                       verbose=5, 

                       model_path=model_path.format(i))



    losses.append(loss)



    # predict

    pred_oof[valid_index] = batch_predict(model, valid_loader_noisy)

    pred_public += batch_predict(model, public_loader) / NFOLDS

    pred_private += batch_predict(model, private_loader) / NFOLDS



    print(f'elapsed: {time.time() - s:.1f}sec')



oof_score = calc_loss(y, pred_oof)

print(f'oof(all): {oof_score: .4f}')



oof_score = calc_loss(y[sn_mask], pred_oof[sn_mask])

print(f'oof(clean): {oof_score: .4f}')



# make submission and oof

preds_ls = []



for df, preds in [(public_df, pred_public), (private_df, pred_private)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=PRED_COLS)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)

preds_df.head()



submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)



np.save('oof', pred_oof)

np.save('public', pred_public)

np.save('private', pred_private)



print(losses)