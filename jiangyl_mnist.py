# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!nvidia-smi
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
import torch

import torch.nn as nn

from collections import OrderedDict

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
print(torch.__version__)
model = nn.Sequential(OrderedDict([

    ('conv_1', nn.Conv2d(1, 16, 3, 2, 0, bias=False)),  # 14,14,16

    ('bn_1', nn.BatchNorm2d(16)),

    ('relu_1', nn.ReLU(inplace=True)),

    ('conv_11', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),  # 14,14,32

    ('bn_11', nn.BatchNorm2d(32)),

    ('relu_11', nn.ReLU(inplace=True)),

    ('conv_12', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),  # 14,14,64

    ('bn_12', nn.BatchNorm2d(64)),

    ('relu_12', nn.ReLU(inplace=True)),



    ('conv_2', nn.Conv2d(64, 128, 3, 2, 0, bias=False)),  # 6,6,128

    ('bn_2', nn.BatchNorm2d(128)),

    ('relu_2', nn.ReLU(inplace=True)),

    ('conv_21', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),  # 6,6,256

    ('bn_21', nn.BatchNorm2d(256)),

    ('relu_21', nn.ReLU(inplace=True)),

    ('conv_22', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),  # 6,6,512

    ('bn_22', nn.BatchNorm2d(512)),

    ('relu_22', nn.ReLU(inplace=True)),



    ('conv_3', nn.Conv2d(512, 1024, 3, 2, 0, bias=False)),  # 2,2,1024

    ('bn_3', nn.BatchNorm2d(1024)),

    ('relu_3', nn.ReLU(inplace=True)),

    ('conv_31', nn.Conv2d(1024, 2048, 1, 1, 0, bias=False)),  # 2,2,2048

    ('bn_31', nn.BatchNorm2d(2048)),

    ('relu_31', nn.ReLU(inplace=True)),

    ('conv_32', nn.Conv2d(2048, 1024, 1, 1, 0, bias=False)),  # 2,2,1024

    ('bn_32', nn.BatchNorm2d(1024)),

    ('relu_32', nn.ReLU(inplace=True)),



    ('avg_pool', nn.AvgPool2d(2, 2)),  # 1,1,1024

    ('flatten', nn.Flatten()),  # 1, 1024

    ('linear_1', nn.Linear(1024, 512)), # 1, 512

    ('relu_linear1', nn.ReLU(inplace=True)), 

    ('drop_out', nn.Dropout(p=0.2)), 

    ('linear_2', nn.Linear(512, 10)), # 1, 10

    ('softmax', nn.Softmax(dim=-1))

]))
def _initialize_weights(m):

    print('initialization ...')

    if isinstance(m, nn.Conv2d):

        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

        m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:

            m.bias.data.zero_()

    elif isinstance(m, nn.BatchNorm2d):

        m.weight.data.fill_(1)

        m.bias.data.zero_()

    elif isinstance(m, nn.Linear):

        m.weight.data.normal_(0, 0.01)

        m.bias.data.zero_()

        

        

model.apply(_initialize_weights)
dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

labels = dataset.iloc[:, 0].values

digits = dataset.iloc[:, 1:].values.reshape(-1, 28, 28)

digits = digits[:, None, :, :]



shuffle = np.random.permutation(len(dataset))

digits = digits[shuffle]

labels = labels[shuffle]

print(f"digits shape: {digits.shape}, \nlabels shape: {labels.shape}")



from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(digits, labels, test_size=0.3)
loss_fn = nn.CrossEntropyLoss()

optimizer_adam = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)

optimizer_sgd = torch.optim.SGD(params=model.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9)

lr_schduler = ExponentialLR(optimizer_sgd, gamma=0.9)

loss_ = float('inf')

loss = torch.tensor(float('inf')).float()
optimizer_adam.param_groups[0]['lr']
model = model.to('cuda')

if Path('./model_every_200.pkl').exists():

    print(f'load pretrained model: {Path("./model_every_200.pkl").resolve()}')

    state_dict = torch.load('./model_best.pkl', map_location='cuda')

#     del state_dict['model']['classifier.weight'], state_dict['model']['classifier.bias']

    model.load_state_dict(state_dict['model'])

#     try:

#         optimizer.load_state_dict(state_dict['optimizer'])

#     except:

#         pass

    epoch = state_dict['epoch']

    loss__ = state_dict['loss']

    print(f'epoch {epoch}; loss {loss__}')

else:

    print('training from stratch ...')

    epoch = 0
random_id = np.random.randint(0, train_x.shape[0], 100)

random_digits_pixel = train_x[random_id]

random_digits_label = train_y[random_id]

random_digits_pixel.shape



hundred_digits = np.zeros(shape=[280, 280], dtype=np.float32)



for i, digit in enumerate(random_digits_pixel):

    row = ((i // 10) + 1)

    col = ((i % 10) + 1)

    hundred_digits[(row-1)*28:row*28, (col-1)*28:col*28] = digit.reshape(28, 28)



plt.figure(figsize=[10,10])

plt.imshow(hundred_digits, cmap='gray')

plt.axis('off')

plt.show()



print(random_digits_label.reshape(10, 10))
def train(batch_x, batch_y, model, loss_fn, optimizer, epoch):

    model.train()

    batch_x = np.stack(batch_x, axis=0)

    batch_x = torch.from_numpy(batch_x).float().to('cuda')

    batch_y = torch.from_numpy(np.array(batch_y)).to(torch.int64).to('cuda')

    pred = model(batch_x)

    loss = loss_fn(pred, batch_y)

    loss_item = loss.detach().cpu().item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f'epoch: {epoch:3d}, step: {i:5d}, train loss: {loss.detach().cpu().item():.6f}')

#     batch_x, batch_y, pred, loss = batch_x.cpu(), batch_y.cpu(), pred.cpu(), loss.cpu()

    del batch_x, batch_y, pred, loss

    return loss_item

    

from sklearn.metrics import accuracy_score

def test(batch_x, batch_y, model, loss_fn, lr):

    model.eval()

    with torch.no_grad():

        batch_x = np.stack(batch_x, axis=0)

        batch_x = torch.from_numpy(batch_x).float().to('cuda')

        batch_y = torch.from_numpy(np.array(batch_y)).to(torch.int64).to('cuda')

        pred = model(batch_x)

        loss = loss_fn(pred, batch_y)

        pred = pred.argmax(dim=-1).squeeze()

        acc = accuracy_score(batch_y.detach().cpu().numpy(), pred.detach().cpu().numpy())

        print(f'test loss {loss}, accuracy {acc}, learning rate {lr}')

#         batch_x, batch_y, pred, loss = batch_x.cpu(), batch_y.cpu(), pred.cpu(), loss.cpu()

        del batch_x, batch_y, pred, loss, acc

    return 
from time import time

torch.backends.cudnn.benchmark = True 

batch_x, batch_y = [], []

epoch=0

start_time = time()

for epoch in range(epoch, 300):

    if epoch < 150:

        optimizer = optimizer_adam

    for i, (x, y) in tqdm(enumerate(zip(train_x, train_y))):

        if (i+1) % 65 != 0:

            batch_x.append(x)

            batch_y.append(y)

        else:

            loss = train(batch_x, batch_y, model, loss_fn, optimizer, epoch)

            batch_x, batch_y = [], []

        if loss_ > loss:

            state_dict = {

                'model': model.state_dict(),

                'optimizer': optimizer.state_dict(),

                'loss': loss,

                'epoch': epoch}

            print('save model best...')

            torch.save(state_dict, './model_best.pkl')

            loss_ = loss

            

        if (i+1) % 500 == 0:

            state_dict = {

                'model': model.state_dict(),

                'optimizer': optimizer.state_dict(),

                'loss': loss,

                'epoch': epoch}

            print('save model every...')

            torch.save(state_dict, './model_every_200.pkl')

            

        if (i + 1) % 300 == 0:

            random_idx = np.random.randint(0, len(test_x), 256)

            lr = optimizer.param_groups[0]['lr']

            test(test_x[random_idx], test_y[random_idx], model, loss_fn, lr)



    if epoch >= 150 and epoch % 10 == 0:

        optimizer = optimizer_sgd

        print(f'learning rate: {lr_schduler.get_lr()[0]}')

        lr_schduler.step()

        

end_time = time()

print(f'total_time:{((end_time - start_time)/60):7.3f} minutes')