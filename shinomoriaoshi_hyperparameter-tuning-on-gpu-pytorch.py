import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

warnings.filterwarnings("ignore")



import gc

import numpy as np

import pandas as pd

import json



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, TensorDataset, DataLoader



from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

from sklearn.preprocessing import QuantileTransformer



from hyperopt import hp, fmin, atpe, tpe, Trials

from hyperopt.pyll.base import scope

from tqdm.notebook import tqdm



import warnings

warnings.filterwarnings('ignore')
import sys

sys.path.append('../input/tabnetdevelop/tabnet-develop')

from pytorch_tabnet.tab_model import TabNetRegressor
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']



train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True)

train = train.loc[train['cp_type']==0].reset_index(drop=True)



features = list(train_features.columns)
GENES = [col for col in train_features.columns if col.startswith('g-')]

CELLS = [col for col in train_features.columns if col.startswith('c-')]



for col in (GENES + CELLS):



    transformer = QuantileTransformer(n_quantiles = 100, random_state = 0, output_distribution="normal")

    vec_len = len(train[col].values)

    vec_len_test = len(test[col].values)

    raw_vec = train[col].values.reshape(vec_len, 1)

    transformer.fit(raw_vec)



    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
def pca(xtrain, features):

    train_features = pd.DataFrame(xtrain, columns = features)

    gene_features = [i for i, x in enumerate(features) if x[:2] == 'g-']

    cell_features = [i for i, x in enumerate(features) if x[:2] == 'c-']



    # For generic features

    pca_g = PCA(n_components = len(gene_features))

    g_pca = pca_g.fit(train_features.iloc[:,gene_features])



    # For cellular features

    pca_c = PCA(n_components = len(cell_features))

    c_pca = pca_c.fit(train_features.iloc[:,cell_features])



    # Now, we care only on 95% of total variance

    cut_g = np.where(np.cumsum(pca_g.explained_variance_ratio_) < 0.80)[0][-1]

    cut_c = np.where(np.cumsum(pca_c.explained_variance_ratio_) < 0.90)[0][-1]

    

    return pca_g, pca_c, gene_features, cell_features, cut_g, cut_c
# Read the json file of important features

with open('../input/t-test-pca-rfe-logistic-regression/main_predictors.json') as json_file:

    important_feats = json.load(json_file)

    

# Extract the important features from the train data

train_top_feats = train[important_feats['start_predictors']]

test_top_feats = test[important_feats['start_predictors']]
def mean_log_loss(y_true, y_pred):

    metrics = []

    for i, target in enumerate(targets):

        metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float), labels = [0,1]))

    return np.mean(metrics)
## TABNET

#from pytorch_tabnet.tab_model import TabModel

import torch

import numpy as np

from scipy.sparse import csc_matrix

import time

from abc import abstractmethod

from pytorch_tabnet import tab_network

from pytorch_tabnet.multiclass_utils import unique_labels

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

from torch.nn.utils import clip_grad_norm_

from pytorch_tabnet.utils import (PredictDataset,

                                  create_dataloaders,

                                  create_explain_matrix)

from sklearn.base import BaseEstimator

from torch.utils.data import DataLoader

from copy import deepcopy

import io

import json

from pathlib import Path

import shutil

import zipfile





class TabModel(BaseEstimator):

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,

                 n_independent=2, n_shared=2, epsilon=1e-15,  momentum=0.02,

                 lambda_sparse=1e-3, seed=0,

                 clip_value=1, verbose=1,

                 optimizer_fn=torch.optim.Adam,

                 optimizer_params=dict(lr=2e-2),

                 scheduler_params=None, scheduler_fn=None,

                 mask_type="sparsemax",

                 input_dim=None, output_dim=None,

                 device_name='auto'):

        """ Class for TabNet model

        Parameters

        ----------

            device_name: str

                'cuda' if running on GPU, 'cpu' if not, 'auto' to autodetect

        """



        self.n_d = n_d

        self.n_a = n_a

        self.n_steps = n_steps

        self.gamma = gamma

        self.cat_idxs = cat_idxs

        self.cat_dims = cat_dims

        self.cat_emb_dim = cat_emb_dim

        self.n_independent = n_independent

        self.n_shared = n_shared

        self.epsilon = epsilon

        self.momentum = momentum

        self.lambda_sparse = lambda_sparse

        self.clip_value = clip_value

        self.verbose = verbose

        self.optimizer_fn = optimizer_fn

        self.optimizer_params = optimizer_params

        self.device_name = device_name

        self.scheduler_params = scheduler_params

        self.scheduler_fn = scheduler_fn

        self.mask_type = mask_type

        self.input_dim = input_dim

        self.output_dim = output_dim



        self.batch_size = 1024



        self.seed = seed

        torch.manual_seed(self.seed)

        # Defining device

        if device_name == 'auto':

            if torch.cuda.is_available():

                device_name = 'cuda'

            else:

                device_name = 'cpu'

        self.device = torch.device(device_name)

        print(f"Device used : {self.device}")



    @abstractmethod

    def construct_loaders(self, X_train, y_train, X_valid, y_valid,

                          weights, batch_size, num_workers, drop_last):

        """

        Returns

        -------

        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader

            Training and validation dataloaders

        -------

        """

        raise NotImplementedError('users must define construct_loaders to use this base class')



    def init_network(

                     self,

                     input_dim,

                     output_dim,

                     n_d,

                     n_a,

                     n_steps,

                     gamma,

                     cat_idxs,

                     cat_dims,

                     cat_emb_dim,

                     n_independent,

                     n_shared,

                     epsilon,

                     virtual_batch_size,

                     momentum,

                     device_name,

                     mask_type,

                     ):

        self.network = tab_network.TabNet(

            input_dim,

            output_dim,

            n_d=n_d,

            n_a=n_a,

            n_steps=n_steps,

            gamma=gamma,

            cat_idxs=cat_idxs,

            cat_dims=cat_dims,

            cat_emb_dim=cat_emb_dim,

            n_independent=n_independent,

            n_shared=n_shared,

            epsilon=epsilon,

            virtual_batch_size=virtual_batch_size,

            momentum=momentum,

            device_name=device_name,

            mask_type=mask_type).to(self.device)



        self.reducing_matrix = create_explain_matrix(

            self.network.input_dim,

            self.network.cat_emb_dim,

            self.network.cat_idxs,

            self.network.post_embed_dim)



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, loss_fn=None,

            weights=0, max_epochs=100, patience=10, batch_size=1024,

            virtual_batch_size=128, num_workers=0, drop_last=False):

        """Train a neural network stored in self.network

        Using train_dataloader for training data and

        valid_dataloader for validation.

        Parameters

        ----------

            X_train: np.ndarray

                Train set

            y_train : np.array

                Train targets

            X_train: np.ndarray

                Train set

            y_train : np.array

                Train targets

            weights : bool or dictionnary

                0 for no balancing

                1 for automated balancing

                dict for custom weights per class

            max_epochs : int

                Maximum number of epochs during training

            patience : int

                Number of consecutive non improving epoch before early stopping

            batch_size : int

                Training batch size

            virtual_batch_size : int

                Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)

            num_workers : int

                Number of workers used in torch.utils.data.DataLoader

            drop_last : bool

                Whether to drop last batch during training

        """

        # update model name



        self.update_fit_params(X_train, y_train, X_valid, y_valid, loss_fn,

                               weights, max_epochs, patience, batch_size,

                               virtual_batch_size, num_workers, drop_last)



        train_dataloader, valid_dataloader = self.construct_loaders(X_train,

                                                                    y_train,

                                                                    X_valid,

                                                                    y_valid,

                                                                    self.updated_weights,

                                                                    self.batch_size,

                                                                    self.num_workers,

                                                                    self.drop_last)



        self.init_network(

            input_dim=self.input_dim,

            output_dim=self.output_dim,

            n_d=self.n_d,

            n_a=self.n_a,

            n_steps=self.n_steps,

            gamma=self.gamma,

            cat_idxs=self.cat_idxs,

            cat_dims=self.cat_dims,

            cat_emb_dim=self.cat_emb_dim,

            n_independent=self.n_independent,

            n_shared=self.n_shared,

            epsilon=self.epsilon,

            virtual_batch_size=self.virtual_batch_size,

            momentum=self.momentum,

            device_name=self.device_name,

            mask_type=self.mask_type

        )



        self.optimizer = self.optimizer_fn(self.network.parameters(),

                                           **self.optimizer_params)



        if self.scheduler_fn:

            self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)

        else:

            self.scheduler = None



        self.losses_train = []

        self.losses_valid = []

        self.learning_rates = []

        self.metrics_train = []

        self.metrics_valid = []



        if self.verbose > 0:

            print("Will train until validation stopping metric",

                  f"hasn't improved in {self.patience} rounds.")

            msg_epoch = f'| EPOCH |  train  |   valid  | total time (s)'

            print('---------------------------------------')

            print(msg_epoch)



        total_time = 0

        while (self.epoch < self.max_epochs and

               self.patience_counter < self.patience):

            starting_time = time.time()

            # updates learning rate history

            self.learning_rates.append(self.optimizer.param_groups[-1]["lr"])



            fit_metrics = self.fit_epoch(train_dataloader, valid_dataloader)



            # leaving it here, may be used for callbacks later

            self.losses_train.append(fit_metrics['train']['loss_avg'])

            self.losses_valid.append(fit_metrics['valid']['total_loss'])

            self.metrics_train.append(fit_metrics['train']['stopping_loss'])

            self.metrics_valid.append(fit_metrics['valid']['stopping_loss'])



            stopping_loss = fit_metrics['valid']['stopping_loss']

            if stopping_loss < self.best_cost:

                self.best_cost = stopping_loss

                self.patience_counter = 0

                # Saving model

                self.best_network = deepcopy(self.network)

                has_improved = True

            else:

                self.patience_counter += 1

                has_improved=False

            self.epoch += 1

            total_time += time.time() - starting_time

            if self.verbose > 0:

                if self.epoch % self.verbose == 0:

                    separator = "|"

                    msg_epoch = f"| {self.epoch:<5} | "

                    msg_epoch += f" {fit_metrics['train']['stopping_loss']:.5f}"

                    msg_epoch += f' {separator:<2} '

                    msg_epoch += f" {fit_metrics['valid']['stopping_loss']:.5f}"

                    msg_epoch += f' {separator:<2} '

                    msg_epoch += f" {np.round(total_time, 1):<10}"

                    msg_epoch += f" {has_improved}"

                    print(msg_epoch)



        if self.verbose > 0:

            if self.patience_counter == self.patience:

                print(f"Early stopping occured at epoch {self.epoch}")

            print(f"Training done in {total_time:.3f} seconds.")

            print('---------------------------------------')



        self.history = {"train": {"loss": self.losses_train,

                                  "metric": self.metrics_train,

                                  "lr": self.learning_rates},

                        "valid": {"loss": self.losses_valid,

                                  "metric": self.metrics_valid}}

        # load best models post training

        self.load_best_model()



        # compute feature importance once the best model is defined

        self._compute_feature_importances(train_dataloader)



    def save_model(self, path):

        """

        Saving model with two distinct files.

        """

        saved_params = {}

        for key, val in self.get_params().items():

            if isinstance(val, type):

                # Don't save torch specific params

                continue

            else:

                saved_params[key] = val



        # Create folder

        Path(path).mkdir(parents=True, exist_ok=True)



        # Save models params

        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:

            json.dump(saved_params, f)



        # Save state_dict

        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))

        shutil.make_archive(path, 'zip', path)

        shutil.rmtree(path)

        print(f"Successfully saved model at {path}.zip")

        return f"{path}.zip"



    def load_model(self, filepath):



        try:

            with zipfile.ZipFile(filepath) as z:

                with z.open("model_params.json") as f:

                    loaded_params = json.load(f)

                with z.open("network.pt") as f:

                    try:

                        saved_state_dict = torch.load(f)

                    except io.UnsupportedOperation:

                        # In Python <3.7, the returned file object is not seekable (which at least

                        # some versions of PyTorch require) - so we'll try buffering it in to a

                        # BytesIO instead:

                        saved_state_dict = torch.load(io.BytesIO(f.read()))

        except KeyError:

            raise KeyError("Your zip file is missing at least one component")



        self.__init__(**loaded_params)



        self.init_network(

            input_dim=self.input_dim,

            output_dim=self.output_dim,

            n_d=self.n_d,

            n_a=self.n_a,

            n_steps=self.n_steps,

            gamma=self.gamma,

            cat_idxs=self.cat_idxs,

            cat_dims=self.cat_dims,

            cat_emb_dim=self.cat_emb_dim,

            n_independent=self.n_independent,

            n_shared=self.n_shared,

            epsilon=self.epsilon,

            virtual_batch_size=1024,

            momentum=self.momentum,

            device_name=self.device_name,

            mask_type=self.mask_type

        )

        self.network.load_state_dict(saved_state_dict)

        self.network.eval()

        return



    def fit_epoch(self, train_dataloader, valid_dataloader):

        """

        Evaluates and updates network for one epoch.

        Parameters

        ----------

            train_dataloader: a :class: `torch.utils.data.Dataloader`

                DataLoader with train set

            valid_dataloader: a :class: `torch.utils.data.Dataloader`

                DataLoader with valid set

        """

        train_metrics = self.train_epoch(train_dataloader)

        valid_metrics = self.predict_epoch(valid_dataloader)



        fit_metrics = {'train': train_metrics,

                       'valid': valid_metrics}



        return fit_metrics



    @abstractmethod

    def train_epoch(self, train_loader):

        """

        Trains one epoch of the network in self.network

        Parameters

        ----------

            train_loader: a :class: `torch.utils.data.Dataloader`

                DataLoader with train set

        """

        raise NotImplementedError('users must define train_epoch to use this base class')



    @abstractmethod

    def train_batch(self, data, targets):

        """

        Trains one batch of data

        Parameters

        ----------

            data: a :tensor: `torch.tensor`

                Input data

            target: a :tensor: `torch.tensor`

                Target data

        """

        raise NotImplementedError('users must define train_batch to use this base class')



    @abstractmethod

    def predict_epoch(self, loader):

        """

        Validates one epoch of the network in self.network

        Parameters

        ----------

            loader: a :class: `torch.utils.data.Dataloader`

                    DataLoader with validation set

        """

        raise NotImplementedError('users must define predict_epoch to use this base class')



    @abstractmethod

    def predict_batch(self, data, targets):

        """

        Make predictions on a batch (valid)

        Parameters

        ----------

            data: a :tensor: `torch.Tensor`

                Input data

            target: a :tensor: `torch.Tensor`

                Target data

        Returns

        -------

            batch_outs: dict

        """

        raise NotImplementedError('users must define predict_batch to use this base class')



    def load_best_model(self):

        if self.best_network is not None:

            self.network = self.best_network



    @abstractmethod

    def predict(self, X):

        """

        Make predictions on a batch (valid)

        Parameters

        ----------

            data: a :tensor: `torch.Tensor`

                Input data

            target: a :tensor: `torch.Tensor`

                Target data

        Returns

        -------

            predictions: np.array

                Predictions of the regression problem or the last class

        """

        raise NotImplementedError('users must define predict to use this base class')



    def explain(self, X):

        """

        Return local explanation

        Parameters

        ----------

            data: a :tensor: `torch.Tensor`

                Input data

            target: a :tensor: `torch.Tensor`

                Target data

        Returns

        -------

            M_explain: matrix

                Importance per sample, per columns.

            masks: matrix

                Sparse matrix showing attention masks used by network.

        """

        self.network.eval()



        dataloader = DataLoader(PredictDataset(X),

                                batch_size=self.batch_size, shuffle=False)



        for batch_nb, data in enumerate(dataloader):

            data = data.to(self.device).float()



            M_explain, masks = self.network.forward_masks(data)

            for key, value in masks.items():

                masks[key] = csc_matrix.dot(value.cpu().detach().numpy(),

                                            self.reducing_matrix)



            if batch_nb == 0:

                res_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(),

                                             self.reducing_matrix)

                res_masks = masks

            else:

                res_explain = np.vstack([res_explain,

                                         csc_matrix.dot(M_explain.cpu().detach().numpy(),

                                                        self.reducing_matrix)])

                for key, value in masks.items():

                    res_masks[key] = np.vstack([res_masks[key], value])

        return res_explain, res_masks



    def _compute_feature_importances(self, loader):

        self.network.eval()

        feature_importances_ = np.zeros((self.network.post_embed_dim))

        for data, targets in loader:

            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)

            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()



        feature_importances_ = csc_matrix.dot(feature_importances_,

                                              self.reducing_matrix)

        self.feature_importances_ = feature_importances_ / np.sum(feature_importances_)

        

        

class TabNetRegressor(TabModel):



    def construct_loaders(self, X_train, y_train, X_valid, y_valid, weights,

                          batch_size, num_workers, drop_last):

        """

        Returns

        -------

        train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader

            Training and validation dataloaders

        -------

        """

        if isinstance(weights, int):

            if weights == 1:

                raise ValueError("Please provide a list of weights for regression.")

        if isinstance(weights, dict):

            raise ValueError("Please provide a list of weights for regression.")



        train_dataloader, valid_dataloader = create_dataloaders(X_train,

                                                                y_train,

                                                                X_valid,

                                                                y_valid,

                                                                weights,

                                                                batch_size,

                                                                num_workers,

                                                                drop_last)

        return train_dataloader, valid_dataloader



    def update_fit_params(self, X_train, y_train, X_valid, y_valid, loss_fn,

                          weights, max_epochs, patience,

                          batch_size, virtual_batch_size, num_workers, drop_last):



        if loss_fn is None:

            self.loss_fn = torch.nn.functional.mse_loss

        else:

            self.loss_fn = loss_fn



        assert X_train.shape[1] == X_valid.shape[1], "Dimension mismatch X_train X_valid"

        self.input_dim = X_train.shape[1]



        if len(y_train.shape) == 1:

            raise ValueError("""Please apply reshape(-1, 1) to your targets

                                if doing single regression.""")

        assert y_train.shape[1] == y_valid.shape[1], "Dimension mismatch y_train y_valid"

        self.output_dim = y_train.shape[1]



        self.updated_weights = weights



        self.max_epochs = max_epochs

        self.patience = patience

        self.batch_size = batch_size

        self.virtual_batch_size = virtual_batch_size

        # Initialize counters and histories.

        self.patience_counter = 0

        self.epoch = 0

        self.best_cost = np.inf

        self.num_workers = num_workers

        self.drop_last = drop_last



    def train_epoch(self, train_loader):

        """

        Trains one epoch of the network in self.network

        Parameters

        ----------

            train_loader: a :class: `torch.utils.data.Dataloader`

                DataLoader with train set

        """



        self.network.train()

        y_preds = []

        ys = []

        total_loss = 0



        for data, targets in train_loader:

            batch_outs = self.train_batch(data, targets)

            y_preds.append(batch_outs["y_preds"].cpu().detach().numpy())

            ys.append(batch_outs["y"].cpu().detach().numpy())

            total_loss += batch_outs["loss"]



        y_preds = np.vstack(y_preds)

        ys = np.vstack(ys)



        #stopping_loss = mean_squared_error(y_true=ys, y_pred=y_preds)

        stopping_loss = mean_log_loss(ys, torch.sigmoid(torch.as_tensor(y_preds)).numpy()  )

        total_loss = total_loss / len(train_loader)

        epoch_metrics = {'loss_avg': total_loss,

                         'stopping_loss': total_loss,

                         }



        if self.scheduler is not None:

            self.scheduler.step()

        return epoch_metrics



    def train_batch(self, data, targets):

        """

        Trains one batch of data

        Parameters

        ----------

            data: a :tensor: `torch.tensor`

                Input data

            target: a :tensor: `torch.tensor`

                Target data

        """

        self.network.train()

        data = data.to(self.device).float()



        targets = targets.to(self.device).float()

        self.optimizer.zero_grad()



        output, M_loss = self.network(data)



        loss = self.loss_fn(output, targets)

        

        loss -= self.lambda_sparse*M_loss



        loss.backward()

        if self.clip_value:

            clip_grad_norm_(self.network.parameters(), self.clip_value)

        self.optimizer.step()



        loss_value = loss.item()

        batch_outs = {'loss': loss_value,

                      'y_preds': output,

                      'y': targets}

        return batch_outs



    def predict_epoch(self, loader):

        """

        Validates one epoch of the network in self.network

        Parameters

        ----------

            loader: a :class: `torch.utils.data.Dataloader`

                    DataLoader with validation set

        """

        y_preds = []

        ys = []

        self.network.eval()

        total_loss = 0



        for data, targets in loader:

            batch_outs = self.predict_batch(data, targets)

            total_loss += batch_outs["loss"]

            y_preds.append(batch_outs["y_preds"].cpu().detach().numpy())

            ys.append(batch_outs["y"].cpu().detach().numpy())



        y_preds = np.vstack(y_preds)

        ys = np.vstack(ys)



        stopping_loss = mean_log_loss(ys, torch.sigmoid(torch.as_tensor(y_preds)).numpy()  ) #mean_squared_error(y_true=ys, y_pred=y_preds)



        total_loss = total_loss / len(loader)

        epoch_metrics = {'total_loss': total_loss,

                         'stopping_loss': stopping_loss}



        return epoch_metrics



    def predict_batch(self, data, targets):

        """

        Make predictions on a batch (valid)

        Parameters

        ----------

            data: a :tensor: `torch.Tensor`

                Input data

            target: a :tensor: `torch.Tensor`

                Target data

        Returns

        -------

            batch_outs: dict

        """

        self.network.eval()

        data = data.to(self.device).float()

        targets = targets.to(self.device).float()



        output, M_loss = self.network(data)

       

        loss = self.loss_fn(output, targets)

        #print(self.loss_fn, loss)

        loss -= self.lambda_sparse*M_loss

        #print(loss)

        loss_value = loss.item()

        batch_outs = {'loss': loss_value,

                      'y_preds': output,

                      'y': targets}

        return batch_outs



    def predict(self, X):

        """

        Make predictions on a batch (valid)

        Parameters

        ----------

            data: a :tensor: `torch.Tensor`

                Input data

            target: a :tensor: `torch.Tensor`

                Target data

        Returns

        -------

            predictions: np.array

                Predictions of the regression problem

        """

        self.network.eval()

        dataloader = DataLoader(PredictDataset(X),

                                batch_size=self.batch_size, shuffle=False)



        results = []

        for batch_nb, data in enumerate(dataloader):

            data = data.to(self.device).float()



            output, M_loss = self.network(data)

            predictions = output.cpu().detach().numpy()

            results.append(predictions)

        res = np.vstack(results)

        return res
# dataset class

class MoaDataset(Dataset):

    def __init__(self, df, targets, feats_idx, mode='train'):

        self.mode = mode

        self.feats = feats_idx

        self.data = df[:, feats_idx]

        if mode=='train':

            self.targets = targets

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        if self.mode == 'train':

            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])

        elif self.mode == 'test':

            return torch.FloatTensor(self.data[idx]), 0
nfolds = 7

nepochs = 150

batch_size = 1024

ntargets = train_targets.shape[1]

targets = [col for col in train_targets.columns]

criterion = nn.BCELoss()
def optimise(params):

    # Define the device

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    

    # Firstly, with these settings of hyperparameters, train the model and store the best model

    print(params)

    print('=' * 50)

    

    res_nn = train_targets.copy()

    res_nn.loc[:, train_targets.columns] = 0

    

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits = nfolds, 

                                       random_state = 0, 

                                       shuffle = True).split(train, train_targets)):

        xtrain, xval = train_top_feats.values[tr], train_top_feats.values[te]

        ytrain, yval = train_targets.values[tr], train_targets.values[te]

        

        model = TabNetRegressor(n_d = params['n'], n_a = params['n'], n_steps = params['n_steps'], gamma = params['gamma'], lambda_sparse = 0, cat_emb_dim = 1,

                                cat_idxs = [], optimizer_fn = optim.Adam, optimizer_params = dict(lr = params['lr'], weight_decay = params['weight_decay']),

                                mask_type = params['mask_type'], device_name = device, verbose = 1, scheduler_params = dict(milestones = [50, 100], gamma = 0.5),

                                scheduler_fn = optim.lr_scheduler.MultiStepLR)

        

        model.fit(X_train = xtrain, y_train = ytrain, X_valid = xval, y_valid = yval, max_epochs = nepochs,

                  patience = 50, batch_size = batch_size, virtual_batch_size = 128, num_workers = 0, drop_last = False,

                  loss_fn = F.binary_cross_entropy_with_logits)

        

        # Inference

        model.load_best_model()

        

        # In the validation set

        preds = model.predict(xval)

        res_nn.loc[te, train_targets.columns] += torch.sigmoid(torch.as_tensor(preds)).detach().cpu().numpy()

        

    res_nn.loc[train_features['cp_type'] == 1, train_targets.columns] = 0

    metrics = []

    for _target in train_targets.columns:

        metrics.append(log_loss(train_targets.loc[:, _target], res_nn.loc[:, _target]))

    print(f'OOF Metric with postprocessing: {np.mean(metrics)}')

    print('=' * 50)

    

    return np.mean(metrics)
param_space = {

    'n': hp.choice('n', [8, 24]),

    'n_steps': hp.choice('n_steps', [1, 2, 3]),

    'gamma': hp.uniform('gamma', 1.0, 2.0),

    'lr': hp.uniform('lr', 0.007, 0.02),

    'weight_decay': hp.uniform('weight_decay', 1e-6, 1e-5),

    'mask_type': hp.choice('mask_type', ['sparsemax', 'entmax'])

}



trials = Trials()



hopt = fmin(fn = optimise,

            space = param_space,

            algo = tpe.suggest, 

            max_evals = 15, 

            timeout = 8.9 * 60 * 60, 

            trials = trials,

           )
print(hopt)