import numpy as np

import pandas as pd

from tqdm.auto import tqdm



import torch

from torch import nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, random_split

from catalyst import dl



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



DATA_DIRECTORY = f"/kaggle/input/{os.listdir('/kaggle/input')[0]}"

RANDOM_SEED = 1235



def file_path(filename):

    global DATA_DIRECTORY

    return os.path.join(DATA_DIRECTORY, filename)
train = pd.read_csv(file_path("train_features.csv")).sort_values(by='sig_id')

targets = pd.read_csv(file_path("train_targets_scored.csv")).sort_values(by='sig_id')

test = pd.read_csv(file_path("test_features.csv"))

submission = test[['sig_id']].assign(**targets.iloc[:, 1:].mean())
mask = test['cp_type'] != 'ctl_vehicle'

submission.iloc[~mask, 1:] = 0
def basic_preprocess(X, y=None):

    mask = X['cp_type'] != 'ctl_vehicle'

    X = X[mask]

    if y is not None:

        y = y[mask].drop(columns='sig_id')

    X.drop(columns=['cp_type', 'sig_id'], inplace=True)

    X['cp_dose'] = ((X['cp_dose'] == 'D2').astype(np.int) - 0.5) * np.sqrt(12)

    X['cp_time1'] = ((X['cp_time'] == 24).astype(np.int) - 0.5) * np.sqrt(12)

    X['cp_time2'] = ((X['cp_time'] == 48).astype(np.int) - 0.5) * np.sqrt(12)

    X.drop(columns='cp_time', inplace=True)

    if y is not None:

        return X, y

    return X



def preprocess(train, targets, test):

    train, targets = basic_preprocess(train, targets)

    test = basic_preprocess(test)

    return train, targets, test



train, targets, test = preprocess(train, targets, test)
X = torch.tensor(train.values, dtype=torch.float)

y = torch.tensor(targets.values, dtype=torch.float)

X_t = torch.tensor(test.values, dtype=torch.float)



dataset = TensorDataset(X, y)

test_dataset = TensorDataset(X_t, torch.zeros(X_t.shape[0], y.shape[1], dtype=y.dtype))
hidden_size = 1024



model = nn.Sequential(

    nn.Linear(X.shape[1], hidden_size),

    nn.ReLU(inplace=True),

    nn.Linear(hidden_size, y.shape[1]),

)



valid_size = int(0.1 * X.shape[0])

train_size = X.shape[0] - valid_size



train_dataset, valid_dataset = random_split(dataset, lengths=[train_size, valid_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

loaders = {

    "train": DataLoader(train_dataset, batch_size=32, shuffle=True),

    "valid": DataLoader(valid_dataset, batch_size=256, shuffle=False)

}

full_loader = {

    "train": DataLoader(dataset, batch_size=32, shuffle=True),

}



optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 11
class CustomRunner(dl.SupervisedRunner):

    loss = nn.BCEWithLogitsLoss(reduction='mean')

    

    def _handle_batch(self, batch):

        y_pred = self.model(batch['features'])



        loss = CustomRunner.loss(y_pred, batch['targets'])

        self.batch_metrics.update({"loss": loss})



runner = CustomRunner()



runner.train(

    model=model,

    optimizer=optimizer,

    loaders=full_loader,

    num_epochs=num_epochs

)
runner = dl.SupervisedRunner()

results = runner.predict_loader(

    model=model,

    loader=DataLoader(test_dataset, batch_size=128)

)



total_results = []

for x in results:

    total_results.append(torch.sigmoid(x['logits']))

total_results = torch.cat(total_results)
submission.iloc[np.where(mask)[0], 1:] = total_results
submission.to_csv("submission.csv", index=None)