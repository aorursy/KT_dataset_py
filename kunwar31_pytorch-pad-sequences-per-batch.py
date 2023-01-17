import torch

from torch.utils import data

import numpy as np

from keras.preprocessing import sequence
class TextDataset(data.Dataset):

    '''

    Simple Dataset

    '''

    def __init__(self,X,y=None):

        self.X = X

        self.y = y



    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return [self.X[idx],self.y[idx]]

        return self.X[idx]
class MyCollator(object):

    '''

    Yields a batch from a list of Items

    Args:

    test : Set True when using with test data loader. Defaults to False

    percentile : Trim sequences by this percentile

    '''

    def __init__(self,test=False,percentile=100):

        self.test = test

        self.percentile = percentile

    def __call__(self, batch):

        if not self.test:

            data = [item[0] for item in batch]

            target = [item[1] for item in batch]

        else:

            data = batch

        lens = [len(x) for x in data]

        max_len = np.percentile(lens,self.percentile)

        data = sequence.pad_sequences(data,maxlen=int(max_len))

        data = torch.tensor(data,dtype=torch.long)

        if not self.test:

            target = torch.tensor(target,dtype=torch.float32)

            return [data,target]

        return [data]
sample_size = 1024

sizes = np.random.normal(loc=200,scale=50,size=(sample_size,)).astype(np.int32)

X = [np.ones((sizes[i])) for i in range(sample_size)]

Y = np.random.rand(sample_size).round()
sizes.max()
batch_size = 128

dataset = TextDataset(X,Y)

test_dataset = TextDataset(X)
collate = MyCollator(percentile=100)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True ,collate_fn=collate)

for X,Y in loader:

    print(X.shape,Y.shape)
test_collate = MyCollator(test=True,percentile=100)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False , collate_fn=test_collate)

for X in test_loader:

    print(X[0].shape)
collate = MyCollator(percentile=95)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True ,collate_fn=collate)

for X,Y in loader:

    print(X.shape,Y.shape)