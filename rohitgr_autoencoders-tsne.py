import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD



import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

import random



from fastprogress import master_bar, progress_bar

from IPython.display import display



import warnings

warnings.filterwarnings('ignore')
SEED = 7



torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True



np.random.seed(SEED)

random.seed(SEED)
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)

df.head()
# Minor preprocessing

df['Time'] = df['Time'] / 3600 % 24
df['Class'].value_counts(normalize=True)
def plot_scatter(X, y, mode='TSNE', fname='file.png'):

    if mode == 'TSNE':

        X_r = TSNE(n_components=2, random_state=SEED).fit_transform(X)

    elif mode == 'PCA':

        X_r = PCA(n_components=2, random_state=SEED).fit_transform(X)

    elif mode == 'TSVD':

        X_r = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(X)

    else:

        print('[ERROR]: Please select a valid mode')

        return

        

    traces = []

    traces.append(go.Scatter(x=X_r[y == 0, 0], y=X_r[y == 0, 1], mode='markers', showlegend=True, name='Non Fraud'))

    traces.append(go.Scatter(x=X_r[y == 1, 0], y=X_r[y == 1, 1], mode='markers', showlegend=True, name='Fraud'))



    layout = dict(title=f'{mode} plot')

    fig = go.Figure(data=traces, layout=layout)

    py.iplot(fig, filename=fname)
fraud = df.loc[df['Class'] == 1]

non_fraud = df.loc[df['Class'] == 0].sample(3000)



new_df = pd.concat([fraud, non_fraud]).sample(frac=1.).reset_index(drop=True)

y = new_df.pop('Class')
plot_scatter(new_df, y, mode='TSNE', fname='tsne1.png')
plot_scatter(new_df, y, mode='PCA', fname='pca1.png')
plot_scatter(new_df, y, mode='TSVD', fname='tsvd1.png')
def get_dls(data, batch_sz, n_workers, valid_split=0.2):

    d_size = len(data)

    ixs = np.random.permutation(range(d_size))



    split = int(d_size * valid_split)

    train_ixs, valid_ixs = ixs[split:], ixs[:split]



    train_sampler = SubsetRandomSampler(train_ixs)

    valid_sampler = SubsetRandomSampler(valid_ixs)



    # Input and output data should be same

    ds = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(data).float())



    train_dl = DataLoader(ds, batch_sz, sampler=train_sampler, num_workers=n_workers)

    valid_dl = DataLoader(ds, batch_sz, sampler=valid_sampler, num_workers=n_workers)



    return train_dl, valid_dl
def train(epochs, model, train_dl, valid_dl, optimizer, criterion, device):

    model = model.to(device)



    mb = master_bar(range(epochs))

    mb.write(['epoch', 'train loss', 'valid loss'], table=True)



    for ep in mb:

        model.train()

        train_loss = 0.

        for train_X, train_y in progress_bar(train_dl, parent=mb):

            train_X, train_y = train_X.to(device), train_y.to(device)

            train_out = model(train_X)

            loss = criterion(train_out, train_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            mb.child.comment = f'{loss.item():.4f}'



        with torch.no_grad():

            model.eval()

            valid_loss = 0.

            for valid_X, valid_y in progress_bar(valid_dl, parent=mb):

                valid_X, valid_y = valid_X.to(device), valid_y.to(device)

                valid_out = model(valid_X)

                loss = criterion(valid_out, valid_y)

                valid_loss += loss.item()

                mb.child.comment = f'{loss.item():.4f}'



        mb.write([f'{ep+1}', f'{train_loss/len(train_dl):.6f}', f'{valid_loss/len(valid_dl):.6f}'], table=True)
class AutoEncoder(nn.Module):

    def __init__(self, f_in):

        super().__init__()



        self.encoder = nn.Sequential(

            nn.Linear(f_in, 100),

            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(100, 70),

            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(70, 40)

        )

        self.decoder = nn.Sequential(

            nn.ReLU(inplace=True),

            nn.Linear(40, 40),

            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(40, 70),

            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(70, f_in)

        )



    def forward(self, x):

        return self.decoder(self.encoder(x))
EPOCHS = 10

BATCH_SIZE = 512

N_WORKERS = 0



model = AutoEncoder(30)

criterion = F.mse_loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = df.drop('Class', axis=1).values

y = df['Class'].values



X = MinMaxScaler().fit_transform(X)

X_nonfraud = X[y == 0]

X_fraud = X[y == 1]

train_dl, valid_dl = get_dls(X_nonfraud[:5000], BATCH_SIZE, N_WORKERS)
train(EPOCHS, model, train_dl, valid_dl, optimizer, criterion, device)
with torch.no_grad():

    model.eval()

    non_fraud_encoded = model.encoder(torch.from_numpy(X_nonfraud).float().to(device)).cpu().numpy()

    fraud_encoded = model.encoder(torch.from_numpy(X_fraud).float().to(device)).cpu().numpy()



nrows = 3000

sample_encoded_X = np.append(non_fraud_encoded[:nrows], fraud_encoded, axis=0)

sample_encoded_y = np.append(np.zeros(nrows), np.ones(len(fraud_encoded)))
plot_scatter(sample_encoded_X, sample_encoded_y, mode='TSNE', fname='tsne2.png')
plot_scatter(sample_encoded_X, sample_encoded_y, mode='PCA', fname='pca2.png')
plot_scatter(sample_encoded_X, sample_encoded_y, mode='TSVD', fname='tsvd2.png')
def print_metric(model, df, y, scaler=None):

    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, shuffle=True, random_state=SEED, stratify=y)

    mets = [accuracy_score, precision_score, recall_score, f1_score]



    if scaler is not None:

        X_train = scaler.fit_transform(X_train)

        X_val = scaler.transform(X_val)



    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)

    train_probs = model.predict_proba(X_train)[:, 1]

    val_preds = model.predict(X_val)

    val_probs = model.predict_proba(X_val)[:, 1]



    train_met = pd.Series({m.__name__: m(y_train, train_preds) for m in mets})

    train_met['roc_auc'] = roc_auc_score(y_train, train_probs)

    val_met = pd.Series({m.__name__: m(y_val, val_preds) for m in mets})

    val_met['roc_auc'] = roc_auc_score(y_val, val_probs)

    met_df = pd.DataFrame()

    met_df['train'] = train_met

    met_df['valid'] = val_met



    display(met_df)
encoded_X = np.append(non_fraud_encoded, fraud_encoded, axis=0)

encoded_y = np.append(np.zeros(len(non_fraud_encoded)), np.ones(len(fraud_encoded)))
clf = LogisticRegression(random_state=SEED)

print('Metric scores for original data:')

print_metric(clf, X, y)

print('Metric score for encoded data:')

print_metric(clf, encoded_X, encoded_y)