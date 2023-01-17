import pandas as pd

import numpy as np

import os

from sklearn.preprocessing import StandardScaler



data_dir = '../input/creditcardfraud'

model_dir = '../input/credit-card-fraud-detection-models'



# Load the data and define its feature columns.

txns = pd.read_csv(os.path.join(data_dir, 'creditcard.csv'))

feat_cols = [i for i in txns.columns if i[0] == 'V']

feat_cols.extend(['Amount', 'Time'])

print('The feature columns are: {}'.format(feat_cols))



# Normalize the features.

txns[feat_cols] = StandardScaler().fit_transform(txns[feat_cols])



# Build and count the number of records in Fraud and Non-Fraud classes.

non_frauds = txns[txns.Class == 0]

frauds = txns[txns.Class == 1]

print('Total of {} frauds and {} non-frauds'.format(len(frauds), len(non_frauds)))



# For asynchronous training we can use multiple processes, but then the results vary depending

# on the order of completion. 

pool_size = 16
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Draw random samples for visualization.

non_fraud_samples = 15 * len(frauds)

vis_subset = pd.concat([

    frauds, non_frauds.sample(n=non_fraud_samples, random_state=361932)],axis=0)[feat_cols]



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(vis_subset)

tsne_frame = pd.DataFrame({'X': tsne_results[:, 0], 'Y': tsne_results[:, 1]})

tsne_frame['Fraud'] = [ int(x < len(frauds)) for x in range(0, len(tsne_results)) ]



plt.figure(figsize=(16,10))

pal = [(0,1,0), (1,0,0)]

sns.scatterplot(

    data=tsne_frame,

    palette=pal,

    x = 'X', y = 'Y',hue='Fraud',

    style='Fraud')
import torch



from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset



class CreditCardDataset(Dataset):

    def __init__(self, name, X, y):

        assert(len(X) == len(y))

        self.name = name

        if isinstance(X, pd.DataFrame):    

            self.X = torch.tensor(X.values, dtype=torch.float32)

        else:

            self.X = X.clone()

        if isinstance(y, pd.Series):

            self.y = torch.tensor(y.values, dtype=torch.float32)

        else:

            self.y = y.clone()

    

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx):

        return (self.X[idx], self.y[idx])

    

    def __str__(self):

        return 'CreditCardDataset "{}" has {} items and {} frauds'.format(self.name, len(self.y), int(self.y.sum()))

    

X, y = txns[feat_cols], txns['Class']

# Reserve 30% for dev and training

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=3161)

# Split the reserved 30% in half for 15% dev and 15% test

X_dev, X_test, y_dev, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1494)



train_set = CreditCardDataset('train', X_train, y_train)

dev_set = CreditCardDataset('dev', X_dev, y_dev)

test_set = CreditCardDataset('test', X_test, y_test)

print(train_set)

print(dev_set)

print(test_set)
from os import path



from torch.multiprocessing import Pool



def init_weights(m):

    if type(m) == torch.nn.Linear:

        torch.nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0.1)



def train_model_epoch(model, loader, optimizer):

    loss_fn = torch.nn.BCELoss()

    total_loss = 0

    for i, data in enumerate(loader, 0):

        X, y = data

        y_pred = model(X)

        loss = loss_fn(y_pred.reshape(-1), y)

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    return total_loss

        

def train_model(model, loader, optimizer, state_name, max_epochs=50, checkpoints=5, silent=False):

    state_file = '{}/{}.pt'.format(model_dir, state_name)

    if path.exists(state_file):

        model.load_state_dict(torch.load(state_file))

        return

    # torch.manual_seed(3621099)

    model.train()

    model.apply(init_weights)

    checks, check_at, total_loss = 0, 0, 0

    pool = Pool(processes=pool_size)

    pool_args = [(model, loader, optimizer) for _ in range(max_epochs)]

    for epoch, result in enumerate(pool.starmap_async(train_model_epoch, pool_args).get()):

        total_loss = result

        if not silent and epoch >= check_at:

            print('At epoch {}, loss={}'.format(epoch, total_loss))

            checks += 1

            check_at = int(max_epochs * checks / checkpoints)    

    if not silent:

        print('Final loss={}'.format(total_loss))

    pool.close()

    torch.save(model.state_dict(), state_file)

    return
import matplotlib.pyplot as plt



from sklearn.metrics import average_precision_score, precision_recall_curve

    

%matplotlib inline



def eval_model(model, name, eval_set, batch_size, silent=False):

    model.eval()

    full_loader = torch.utils.data.DataLoader(eval_set, batch_size)

    loss_fn = torch.nn.BCELoss()

    total_loss = 0

    device = next(model.parameters()).device

    y_data_all, y_pred_all = None, None

    for i, data in enumerate(full_loader, 0):

        X, y = data

        y_pred = model(X)

        if i == 0:

            y_data_all = y

            y_pred_all = y_pred

        else:

            y_data_all = torch.cat((y_data_all, y), 0)

            y_pred_all = torch.cat((y_pred_all, y_pred), 0)

        total_loss += loss_fn(y_pred.reshape(-1), y)

    y_data_all = y_data_all.detach().numpy()

    y_pred_all = y_pred_all.detach().numpy()

    avg_precision = average_precision_score(y_data_all, y_pred_all)

    if not silent:

        print('y_data_all sum={}'.format(y_data_all.sum()))

        print('y_pred_all sum={}'.format(y_pred_all.sum()))

        precision, recall, _ = precision_recall_curve(y_data_all, y_pred_all)

        plt.step(recall, precision, alpha=0.3, color='b')

        plt.fill_between(recall, precision, alpha=0.3, color='b')

        plt.xlabel('Recall')

        plt.ylabel('Precision')

        plt.title('{} Precision-Recall, AP={:0.3f}'.format(name, avg_precision))

        plt.show()

        print('Loss of {}={}'.format(name, total_loss.item()))

    return avg_precision
batch_size, d_in, d_out = 300, len(feat_cols), 1

logistic_model = torch.nn.Sequential(

    torch.nn.Linear(d_in, d_out),

    torch.nn.Sigmoid(),

)



if __name__ == '__main__':

    learning_rate = 0.002

    loader = torch.utils.data.DataLoader(train_set, batch_size)

    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=learning_rate, weight_decay=0)

    train_model(model=logistic_model, loader=loader, optimizer=optimizer, state_name='logistic-0', max_epochs=200)

    eval_model(model=logistic_model, name='Logistic model (WD=0)', eval_set=dev_set, batch_size=batch_size)
import numpy as np

from multiprocessing import Pool, cpu_count



def eval_model_for_datasets(model, train_set, eval_sets, batch_size, learning_rate, weight_decay, model_name,

                            max_epochs=200):

    loader = torch.utils.data.DataLoader(train_set, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    state_name = '{}-{:0.6f}'.format(model_name, weight_decay)

    train_model(model=model, loader=loader, optimizer=optimizer, state_name=state_name, 

                max_epochs=max_epochs, silent=True)

    avg_precisions = []

    for i, eval_set in enumerate(eval_sets):

        avg_precisions.append(eval_model(model=model,

                                         name='Dataset #{} (WD={:0.6f})'.format(i + 1, weight_decay),

                                         eval_set=eval_set, batch_size=batch_size, silent=True))

    return tuple([float(weight_decay)]) + tuple(avg_precisions)



def eval_model_over_weight_decays(model, train_set, eval_sets,

                                  batch_size, learning, weight_decays, model_name, max_epochs=200, pool_size=32):

    wd_precisions = dict()

    for wd in weight_decays:

        result = eval_model_for_datasets(model=logistic_model, train_set=train_set,

                                         eval_sets=eval_sets, batch_size=batch_size,

                                         learning_rate=learning_rate, weight_decay=wd,

                                         model_name=model_name, max_epochs=max_epochs)

        wd_precisions[result[0]] = result[1:]



    wd_and_precisions_len = 1 + len(eval_sets)

    wd_and_precisions = [list() for _ in range(wd_and_precisions_len)]

    for wd in sorted(wd_precisions.keys()):

        wl = list()

        wd_and_precisions[0].append(wd)

        for i, prec in enumerate(wd_precisions[wd]):

            wd_and_precisions[i + 1].append(prec)

    plt.xlabel('Weight Decay')

    plt.ylabel('Average Precision')

    for i, dataset in enumerate(eval_sets):

        plt.plot(wd_and_precisions[0], wd_and_precisions[i+1], color='C{}'.format(i), 

                 label='{} set'.format(dataset.name))

    plt.legend(loc='best')

    plt.show()
import numpy as np



%matplotlib inline



if __name__ == '__main__':

    batch_size, learning_rate = 300, 0.002

    eval_model_over_weight_decays(logistic_model, train_set, [train_set, dev_set], batch_size, learning_rate,

                                  np.geomspace(0.00001, 0.8, 20), 'logistic')
batch_size, d_in, d_h1, d_h2, d_out = 60, len(feat_cols), len(feat_cols), len(feat_cols), 1

nn_model = torch.nn.Sequential(

    torch.nn.Linear(d_in, d_h1),

    torch.nn.ReLU(),

    torch.nn.Linear(d_h1, d_h2),

    torch.nn.ReLU(),

    torch.nn.Linear(d_h2, d_out),  

    torch.nn.Sigmoid(),

)



if __name__ == '__main__':

    learning_rate = 0.002   

    loader = torch.utils.data.DataLoader(train_set, batch_size)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=0)

    train_model(model=nn_model, loader=loader, optimizer=optimizer, state_name='nn-0', max_epochs=200)

    eval_model(model=nn_model, name='3 Layer model (WD=0)', eval_set=dev_set, batch_size=batch_size)

   
import numpy as np



%matplotlib inline



if __name__ == '__main__':

    batch_size, learning_rate = 300, 0.002

    eval_model_over_weight_decays(nn_model, train_set, [train_set, dev_set], batch_size, learning_rate,

                                  np.geomspace(0.00001, 0.8, 20), 'nn')
import lzma

import pickle

import resource



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_fscore_support



def model_file(model_name):

    return '{}/{}.lzma.pkl'.format(model_dir, model_name)

    

def load_model(model_name):

    file = model_file(model_name)

    if not path.exists(model_file(model_name)):

        return None

    with lzma.open(file, 'rb') as file:

        return pickle.load(file)

        

def save_model(model_name, model):

    with lzma.open(model_file(model_name), 'wb') as fh:

        pickle.dump(model, fh)



def eval_pred(name, y_data, y_pred, fscore=True):

    precision, recall, _ = precision_recall_curve(y_data, y_pred) 

    avg_precision = average_precision_score(y_data, y_pred)

    plt.step(recall, precision, alpha=0.3, color='b')

    plt.fill_between(recall, precision, alpha=0.3, color='b')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('{}, AP={:0.3f}, '.format(name, avg_precision))

    plt.show()

    if fscore:

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_data, y_pred, beta=1.0, average='binary')

        print('{} Precision={:0.4f}, Recall={:0.4f}, F1={:0.4f}'.format(name, precision, recall, f1_score)) 

    

def eval_knn(model, name, eval_set):

    y_pred = model.predict(eval_set.X)

    eval_pred(name, eval_set.y, y_pred)
for n in [1,3,4,5,10]:

    model_name = 'knn-{}'.format(n)

    knn = load_model(model_name)

    if not knn:

        knn = KNeighborsClassifier(n_neighbors=n, algorithm='kd_tree', leaf_size=50)

        # On a machine with low memory I found I needed to train on a subset of the full data.

        # knn.fit(train_set.X[0:5000, :], train_set.y[0:5000])

        knn.fit(train_set.X, train_set.y)

        save_model(model_name, knn)

    eval_knn(knn, 'k-NN dev (N={})'.format(n), dev_set)




def cluster_probs(clusterer):

    p, q, cluster_prob = dict(), dict(), dict()

    for i, k in enumerate(clusterer.labels_):

        if k not in p:

            p[k] = 0

            q[k] = 0

        q[k] += 1

        if train_set.y[i] == 1:

            p[k] += 1

    for k in p:

        cluster_prob[k] = float(p[k] / q[k])

    return cluster_prob



def eval_hdbscan(name, clusterer, eval_set):

    test_labels, _ = hdbscan.approximate_predict(clusterer, eval_set.X)

    probs = cluster_probs(clusterer)

    y_pred = list()

    ks = 0

    for label in test_labels:

        kp = probs[label]

        ks += kp

        y_pred.append(kp)

    eval_pred(name, eval_set.y, y_pred, fscore=False)
!pip install hdbscan

import hdbscan



clusterer = load_model('hdbscan-2-1')

if not clusterer:

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)

    clusterer.fit_predict(train_set.X)

    save_model('hdbscan-2-1', clusterer)

eval_hdbscan('HDBSCAN min_cluster_size=2, min_samples=1', clusterer, dev_set)
train_model(model=logistic_model, loader=None, optimizer=None, state_name='logistic-0')

eval_model(model=logistic_model, name='Logistic model (WD=0)', eval_set=test_set, batch_size=batch_size)

    

train_model(model=nn_model, loader=None, optimizer=None, state_name='nn-0')

eval_model(model=nn_model, name='2 Hidden Layer model (WD=0)', eval_set=test_set, batch_size=batch_size)



knn = load_model('knn-4')

eval_knn(knn, 'k-NN dev (N=4)', test_set)
from torch.utils.data import Sampler

from itertools import chain



import random



class FraudSampler(Sampler):

    def __init__(self, cc_dataset, batch_size=50, frauds=20):

        self.fraud_count, self.non_fraud_count = frauds, batch_size - frauds

        self.non_fraud_idx = (cc_dataset.y == 0).nonzero().squeeze().tolist()

        self.fraud_idx = (cc_dataset.y > 0).nonzero().squeeze().tolist()

        self.block_size = (self.fraud_count + self.non_fraud_count)

        self.blocks = int(len(self.non_fraud_idx) / self.non_fraud_count)



    def __iter__(self):

        iters = list()

        for i in range(self.blocks):

            block = list()

            block.extend(self.non_fraud_idx[i * self.non_fraud_count: (i+1) * self.non_fraud_count])

            block.extend(random.sample(self.fraud_idx, self.fraud_count))

            iters.append(iter(block))

        return chain(*iters)

            

    def __len__(self):

        return self.blocks * self.block_size

    

    

batch_size = 5000

fraud_sampler = FraudSampler(dev_set, batch_size=batch_size, frauds=60)

loader = torch.utils.data.DataLoader(dev_set, batch_size, sampler=fraud_sampler)

for i, data in enumerate(loader, 0):

    X, y = data

    print('Batch {} has {} examples with {} frauds'.format(i, len(y), int(y.sum())))