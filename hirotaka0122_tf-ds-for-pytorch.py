import numpy as np

from sklearn.datasets import make_classification



import torch

from torch.utils.data import DataLoader, Dataset, TensorDataset



import tensorflow as tf

import tensorflow_datasets as tfds
x, y = make_classification(n_samples=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())

torch_loader = DataLoader(torch_ds, batch_size=128)
%%time

for epoch in range(100):

    for batch in torch_loader:

        x_train, y_train = batch

        x_train = x_train.to(device)

        y_train = y_train.to(device)

        
tf_ds = tf.data.Dataset.from_tensor_slices((x, y))

tf_ds_loader = tfds.as_numpy(tf_ds.batch(128))
%%time

for epoch in range(100):

    for batch in tf_ds_loader:

        x_train, y_train = batch

        x_train = torch.tensor(x_train, device=device)

        y_train = torch.tensor(y_train, device=device)

        