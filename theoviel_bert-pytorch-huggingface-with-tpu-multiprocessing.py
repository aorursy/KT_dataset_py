import os

import collections

from datetime import datetime, timedelta



os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470"



_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')

VERSION = "torch_xla==nightly"

CONFIG = {

    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(

        (datetime.today() - timedelta(1)).strftime('%Y%m%d')))}[VERSION]



DIST_BUCKET = 'gs://tpu-pytorch/wheels'

TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)



!export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

!apt-get install libomp5 -y

!apt-get install libopenblas-dev -y



!pip uninstall -y torch torchvision

!gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .

!gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .

!gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .

!pip install "$TORCH_WHEEL"

!pip install "$TORCH_XLA_WHEEL"

!pip install "$TORCHVISION_WHEEL"
import gc

import os

import time

import math

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from datetime import date

from transformers import *

from sklearn.metrics import *

from tqdm.notebook import tqdm



import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F



from torch import Tensor

from torch.optim import *

from torch.nn.modules.loss import *

from torch.optim.lr_scheduler import * 

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import RandomSampler

from torch.utils.data.distributed import DistributedSampler



import torch_xla

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
def seed_everything(seed):

    """

    Seeds basic parameters for reproductibility of results

    

    Arguments:

        seed {int} -- Number of the seed

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
seed = 2020

seed_everything(seed)
MODEL_PATHS = {

    'bert-multi-cased': '../input/bertconfigs/multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12/',

}
DATA_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/'



df_val = pd.read_csv(DATA_PATH + 'validation-processed-seqlen128.csv')

df_test =  pd.read_csv(DATA_PATH + 'test-processed-seqlen128.csv')

df_train = pd.read_csv(DATA_PATH + 'jigsaw-toxic-comment-train-processed-seqlen128.csv')
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)



sns.countplot(df_train['toxic'])

plt.title('Target repartition on training data')



plt.subplot(1, 2, 2)

sns.countplot(df_val['toxic'])

plt.title('Target repartition on validation data')



plt.show()
class JigsawDataset(Dataset):

    """

    Torch dataset for the competition.

    """

    def __init__(self, df):

        """

        Constructor

        

        Arguments:

            df {pandas dataframe} -- Dataframe where the data is. Expects to be one of the []-processed-seqlen128.csv files

        """

            

        super().__init__()

        self.df = df 

        self.word_ids = np.array([word_ids[1:-1].split(', ') for word_ids in df['input_word_ids']]).astype(int)

        

        try:

            self.y = df['toxic'].values

        except KeyError: # test data

            self.y = np.zeros(len(df))



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        return torch.tensor(self.word_ids[idx]), torch.tensor(self.y[idx])
TRANSFORMERS = {

    "bert-multi-cased": (BertModel, BertTokenizer, "bert-base-uncased"),

}
class Transformer(nn.Module):

    def __init__(self, model, num_classes=1):

        """

        Constructor

        

        Arguments:

            model {string} -- Transformer to build the model on. Expects "camembert-base".

            num_classes {int} -- Number of classes (default: {1})

        """

        super().__init__()

        self.name = model



        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]



        bert_config = BertConfig.from_json_file(MODEL_PATHS[model] + 'bert_config.json')

        bert_config.output_hidden_states = True

        

        self.transformer = BertModel(bert_config)



        self.nb_features = self.transformer.pooler.dense.out_features



        self.pooler = nn.Sequential(

            nn.Linear(self.nb_features, self.nb_features), 

            nn.Tanh(),

        )



        self.logit = nn.Linear(self.nb_features, num_classes)



    def forward(self, tokens):

        """

        Usual torch forward function

        

        Arguments:

            tokens {torch tensor} -- Sentence tokens

        

        Returns:

            torch tensor -- Class logits

        """

        _, _, hidden_states = self.transformer(

            tokens, attention_mask=(tokens > 0).long()

        )



        hidden_states = hidden_states[-1][:, 0] # Use the representation of the first token of the last layer



        ft = self.pooler(hidden_states)



        return self.logit(ft)
def fit(model, train_dataset, val_dataset, epochs=1, batch_size=32, warmup_prop=0, lr=5e-5):

    device = xm.xla_device()

    model = model.to(device)

    

    train_sampler = DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )

    

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=batch_size,

        sampler=train_sampler,

        num_workers=0,

        drop_last=True

    )

    

    valid_loader = torch.utils.data.DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        drop_last=True

    )

    

    

    lr = lr * xm.xrt_world_size()

    batch_size = batch_size # * xm.xrt_world_size() # not sure about that

    

    optimizer = AdamW(model.parameters(), lr=lr)

    

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))

    num_training_steps = epochs * len(train_loader)

    

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    

    loss_fct = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    

    def train_loop(train_loader):

        optimizer.zero_grad()

        avg_loss = 0

        

        for step, (x, y_batch) in enumerate(train_loader): 

            y_pred = model(x.to(device))

            

            loss = loss_fct(y_pred.view(-1).float(), y_batch.float().to(device))

            loss.backward()

            avg_loss += loss.item()



            xm.optimizer_step(optimizer, barrier=True)

            scheduler.step()

            model.zero_grad()

            optimizer.zero_grad()        

            

        return avg_loss

            

    def val_loop(val_loader):

        preds = []

        truths = []

        avg_val_loss = 0.



        with torch.no_grad():

            for x, y_batch in val_loader:                

                y_pred = model(x.to(device))

                loss = loss_fct(y_pred.detach().view(-1).float(), y_batch.float().to(device))

                avg_val_loss += loss.item()

                

                probs = torch.sigmoid(y_pred).detach().cpu().numpy()

                preds += list(probs.flatten())

                truths += list(y_batch.cpu().numpy().flatten())

                

        return (preds, truths, avg_val_loss)

        

    for epoch in range(epochs):

        avg_losses = []

        val_loop_out = []

        start_time = time.time()

        

        # Train

        model.train()

        para_loader = pl.ParallelLoader(train_loader, [device])

        avg_losses.append(train_loop(para_loader.per_device_loader(device)))

        

        # Eval

        model.eval()

        para_loader = pl.ParallelLoader(valid_loader, [device])

        val_loop_out.append(val_loop(para_loader.per_device_loader(device)))

        

        # Retrieve outputs

        preds = np.array(np.concatenate([v[0] for v in val_loop_out])).flatten()

        truths = np.array(np.concatenate([v[1] for v in val_loop_out])).flatten()

        avg_val_losses = [v[2] for v in val_loop_out]

        

        # Metrics

        avg_val_loss = np.sum(avg_val_losses) / len(val_dataset)

        avg_loss = np.sum(avg_val_losses) / len(val_dataset)

        val_auc = roc_auc_score(truths, preds)

        

        dt = time.time() - start_time

        lr = scheduler.get_last_lr()[0]

        

        xm.master_print(f'Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={dt:.0f}s \t loss={avg_loss:.4f} \t val_loss={avg_val_loss:.4f} \t val_auc={val_auc:.4f}')

        xm.save(model.state_dict(), "model.pt")
model = Transformer("bert-multi-cased")
epochs = 1 # 1 epoch seems to be enough

batch_size = 32

warmup_prop = 0.1

lr = 2e-5  # Important parameter to tweak
train_dataset = JigsawDataset(df_train)

val_dataset = JigsawDataset(df_val)

test_dataset = JigsawDataset(df_test)
def fit_multiprocessing(rank, flags):

    fit(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size, warmup_prop=warmup_prop, lr=lr)

    

FLAGS = {}

xmp.spawn(fit_multiprocessing, args=(FLAGS,), nprocs=8, start_method='fork')
def load_model_weights(model, filename, verbose=1, strict=True):

    print(f'\n -> Loading weights from {filename}\n')

    model.load_state_dict(torch.load(filename, map_location='cpu'), strict=strict)

    return model
# def predict(model, dataset, batch_size=64):

#     """

#     Usual predict torch function

    

#     Arguments:

#         model {torch model} -- Model to predict with

#         dataset {torch dataset} -- Dataset to get predictions from

    

#     Keyword Arguments:

#         batch_size {int} -- Batch size (default: {32})

    

#     Returns:

#         numpy array -- Predictions

#     """

#     device = xm.xla_device()

#     model = model.to(device)

#     model.eval()

    

#     preds = []

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    

#     def pred_loop(val_loader):

#         preds = []



#         with torch.no_grad():

#             for x, y_batch in val_loader:                

#                 y_pred = model(x.to(device))

#                 preds += list(probs.flatten())

#         return preds



#     preds.append(val_loop(para_loader.per_device_loader(device)))

    



    

#     with torch.no_grad():

#         for x, _ in tqdm(loader):

#             probs = torch.sigmoid(model(x.to(device))).detach().cpu().numpy()

#             preds = np.concatenate([preds, probs])

            

#     return preds
def predict(model, dataset, batch_size=64):

    """

    Usual predict torch function

    

    Arguments:

        model {torch model} -- Model to predict with

        dataset {torch dataset} -- Dataset to get predictions from

    

    Keyword Arguments:

        batch_size {int} -- Batch size (default: {32})

    

    Returns:

        numpy array -- Predictions

    """

    

    device = xm.xla_device()

    model = model.to(device)

    model.eval()

    

    preds = np.empty((0, 1))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



    with torch.no_grad():

        for x, _ in tqdm(loader):

            probs = torch.sigmoid(model(x.to(device))).detach().cpu().numpy()

            preds = np.concatenate([preds, probs])

            

    return preds
model = load_model_weights(model, 'model.pt')
pred_val = predict(model, val_dataset)

df_val['pred'] = pred_val
for language in df_val['lang'].unique():

    lang_score = roc_auc_score(

        df_val[df_val['lang'] == language]['toxic'], 

        df_val[df_val['lang']  == language]['pred']

    )

    print(f'AUC for language {language}: {lang_score:.4f}')
score = roc_auc_score(df_val['toxic'], pred_val)

print(f'Scored {score:.4f} on validation data')
pred_test = predict(model, test_dataset)
sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

sub['toxic'] = pred_test

sub.to_csv('submission.csv', index=False)

sub.head()