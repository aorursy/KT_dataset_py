# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torchvision as vision

from torch import nn
train_source_frame = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_source_frame = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_label = torch.tensor(train_source_frame.loc[:, 'label'].to_numpy()).reshape(-1).long()

train_features = torch.tensor(train_source_frame.loc[:, tuple(map(lambda idx: f'pixel{idx}', range(train_source_frame.shape[1] - 1)))].to_numpy()).reshape(-1, 1, 28, 28).float()
train_label.shape

train_features.shape
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Loss, Accuracy

from ignite.contrib.handlers.tqdm_logger import ProgressBar
train_ratio = 0.9

batch_size = 2048

learning_rate = 5e-5
class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, features, label):

        assert features.shape[0] == label.shape[0]

        self._features = features

        self._label = label

        

    def __getitem__(self, item):

        return self._features[item], self._label[item]

    

    def __len__(self):

        return self._label.shape[0]



train_slice = slice(0, int(train_label.shape[0] * train_ratio))

valid_slice = slice(int(train_label.shape[0] * train_ratio), None)
train_dataset = MNISTDataset(train_features[train_slice], train_label[train_slice])

valid_dataset = MNISTDataset(train_features[valid_slice], train_label[valid_slice])

len(train_dataset), len(valid_dataset)
train_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=batch_size,

    shuffle=True,

    pin_memory=True

)



valid_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=batch_size,

    shuffle=True,

    pin_memory=True

)
model = nn.Sequential(

    nn.Conv2d(1, 32, 3, bias=False),

    nn.BatchNorm2d(32),

    nn.SELU(inplace=True),

    nn.Conv2d(32, 128, 1, bias=False),

    nn.BatchNorm2d(128),

    nn.SELU(inplace=True),

    nn.Conv2d(128, 192, 5, bias=False),

    nn.BatchNorm2d(192),

    nn.SELU(inplace=True),

    nn.Conv2d(192, 32, 22, bias=False),

    nn.BatchNorm2d(32),

    nn.SELU(inplace=True),

    nn.Conv2d(32, 10, 1, bias=True),

    nn.Flatten(),

)
trainer = create_supervised_trainer(

    model=model,

    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),

    loss_fn=torch.nn.CrossEntropyLoss(),

    device=torch.device('cuda:0')

)

evaluator = create_supervised_evaluator(

    model=model,

    metrics={

        'accuracy': Accuracy(),

        'loss': Loss(torch.nn.CrossEntropyLoss())

    },

    device=torch.device('cuda:0')

)
accuracy = []

loss = []

@trainer.on(Events.EPOCH_COMPLETED)

def on_epoch_completed(engine):

    evaluator.run(valid_loader)

    accuracy.append(evaluator.state.metrics["accuracy"])

    loss.append(evaluator.state.metrics["loss"])

    print(f'Epoch {engine.state.epoch} Accuracy {evaluator.state.metrics["accuracy"]} Loss {evaluator.state.metrics["loss"]}')



pbar = ProgressBar()

pbar.attach(trainer)

trainer.run(train_loader, 50)
from matplotlib import pyplot as plt
plt.plot(accuracy)

plt.plot(loss)
max(accuracy), min(loss)
torch.cuda.empty_cache()
test_features = torch.tensor(test_source_frame.loc[:, tuple(map(lambda idx: f'pixel{idx}', range(test_source_frame.shape[1])))].to_numpy()).reshape(-1, 1, 28, 28).float()

test_dataset = torch.utils.data.TensorDataset(test_features)

test_loader = torch.utils.data.DataLoader(

    test_dataset,

    batch_size=256,

    shuffle=False,

    pin_memory=True

)
list(model.parameters())
import gc

gc.collect()
import tqdm
torch.cuda.empty_cache()

gc.collect()
outputs = []

model.eval()

for batch in tqdm.tqdm(test_loader):

    with torch.no_grad():

        output = model(batch[0].to('cuda:0'))

        batch[0] = batch[0].to('cpu')

        outputs.append(output.to('cpu'))

        del output, batch

        if len(outputs) % 1 == 0:

            torch.cuda.empty_cache()

            gc.collect()

    
target = torch.cat(outputs).argmax(axis=1).numpy()

submission = pd.DataFrame({'ImageId': list(range(1, target.shape[0] + 1)), 'Label': target})
submission.to_csv('/kaggle/working/submission.csv')
torch.save(model, '/kaggle/working/model.pt')