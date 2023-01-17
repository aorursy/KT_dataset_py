%matplotlib inline
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
try:
    from skimage.util.montage import montage2d
except ImportError as e:
    print('scikit-image is too new, ',e)
    from skimage.util import montage as montage2d
# deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
DATA_ROOT_PATH = os.path.join('..', 'input')
!ls -lh {DATA_ROOT_PATH} # show the data
class AffMNISTDataset(Dataset):
    def __init__(self, data_name):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with np.load(os.path.join(DATA_ROOT_PATH, '{}.npz'.format(data_name))) as npz_data:
            self.img_vec = npz_data['img']
            self.idx_vec = npz_data['idx'] # the id for each image so we can match the labels
            print('image shape', self.img_vec.shape)
            print('idx shape', self.idx_vec.shape)
        label_path = os.path.join(DATA_ROOT_PATH, '{}_labels.csv'.format(data_name))
        if os.path.exists(label_path):
            label_df = pd.read_csv(label_path)
            self.lab_dict = dict(zip(label_df['idx'], label_df['label'])) # map idx to label
        else:
            self.lab_dict = {x:x for x in self.idx_vec}

    def __len__(self):
        return len(self.img_vec)

    def __getitem__(self, idx):
        out_label = self.lab_dict[self.idx_vec[idx]]
        out_vec = np.array([out_label], dtype='int')
        img = self.img_vec[idx].astype('float32')
        return img, int(out_label)
train_ds = AffMNISTDataset('train')
fig = plt.figure(figsize=(10, 20))
for i in range(len(train_ds)):
    s_image, s_label = train_ds[i]
    print(i, s_image.shape)
    ax = plt.subplot(4, 1, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}: {}'.format(i, s_label))
    ax.axis('off')
    ax.imshow(s_image)
    if i == 3:
        plt.show()
        break
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(40*40, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 40*40)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
train_ds = AffMNISTDataset('train')
train_loader = DataLoader(train_ds, batch_size=1024,
                        shuffle=True, num_workers=4)
model = SimpleMLP().to(device)
model.train()
optimizer = optim.SGD(model.parameters(), 
                      lr=1e-3)
log_interval = 100
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
model.eval() # put model into evaluate mode
test_ds = AffMNISTDataset('test')
test_loader = DataLoader(test_ds, batch_size=1024,
                        shuffle=True, num_workers=4)
target_out = []
pred_out = []
for batch_idx, (data, target_idx) in enumerate(test_loader):
    data = data.to(device)
    target_idx = target_idx.to('cpu').numpy()
    output = model(data) # put data through the model
    pred = output.to('cpu').data.max(1)[1].numpy() #Index of max probability (prediction)
    target_out += [target_idx]
    pred_out += [pred]
pred_df = pd.DataFrame({'idx': np.concatenate(target_out, 0),
              'label': np.concatenate(pred_out, 0)})
pred_df.to_csv('mlp_predictions.csv', index=False)
pred_df.sample(5)
pred_df['label'].hist()
