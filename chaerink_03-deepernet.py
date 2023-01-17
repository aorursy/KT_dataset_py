import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import librosa

import matplotlib.pyplot as plt

import scipy

from tqdm import tqdm

from glob import glob

from scipy.io import wavfile

import torch

from torch import nn, optim

import torch.nn.functional as F
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is NOT available.  Training on CPU')

else:

    print('CUDA is available!  Training on GPU')
y_data = pd.read_csv('/kaggle/input/dacon-sound-classification/train_answer.csv', index_col=0)



temp = y_data.sort_values(by=list(y_data.columns), ascending=False)



tdx = temp.iloc[::4, ].index



y_data = y_data.reset_index()

y_data = y_data[y_data['id'].isin(tdx)]

y_data = y_data.set_index(keys='id')

print(y_data.index[:10])



def data_loader(files):

    out = []

    i = -1

    for file in tqdm(files):

        i += 1

        if i in tdx:

            fs, data = wavfile.read(file)

            out.append(data)    

        else:

            continue

        if i < 10:

            print(file)

    out = np.array(out)

    return out
x_data = sorted(glob('/kaggle/input/dacon-sound-classification/train/*.wav'))

x_data = data_loader(x_data)

x_data.shape
mels = [librosa.feature.melspectrogram(y=x_data[i].astype(np.float), sr=16000, hop_length=128) for i in range(x_data.shape[0])]

len(mels)
import gc

gc.collect()



mels[0].shape
X = np.array(mels)

X = X.reshape(25000, 1, 128, 126)

y = y_data.values

X.shape, y.shape
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler



class MyDataset(Dataset):

    def __init__(self, data, targets):

        self.data = data

        self.targets = targets



    def __getitem__(self, index):

        x = np.asarray(self.data[index])

        y = np.asarray(self.targets[index])



        return x, y



    def __len__(self):

        return len(self.data)







# from torch.utils.data import TensorDataset, DataLoader



# tensor_x = torch.Tensor(x_data) # transform to torch tensor

# tensor_y = torch.Tensor(y_data)



num_train = len(X)

indices = list(range(num_train))

# np.random.shuffle(indices)

split = int(np.floor(0.2 * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



dataset = MyDataset(X, y)

train_loader = DataLoader(dataset, batch_size=16,

    sampler=train_sampler, num_workers=0)

valid_loader = DataLoader(dataset, batch_size=16, 

    sampler=valid_sampler, num_workers=0)
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        self.pool = nn.AvgPool2d(2,2)

        self.drop = nn.Dropout(0.5)

        self.batch1 = nn.BatchNorm2d(16)

        self.batch2 = nn.BatchNorm2d(32)

        self.batch3 = nn.BatchNorm2d(64)

        self.batch4 = nn.BatchNorm2d(128)

        self.batch5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*3, 30)

        

    def forward(self, x):

        

        x = self.pool(F.gelu(self.batch2(self.conv2(F.relu(self.batch1(self.conv1(x)))))))

        x = self.drop(x)

        x = self.pool(F.gelu(self.batch4(self.conv4(F.gelu(self.batch3(self.conv3(x)))))))

        x = self.drop(x)

        x = self.pool(F.gelu(self.batch5(self.conv6(F.gelu(self.batch5(self.conv5(x)))))))

        x = self.drop(x)

        x = self.pool(F.gelu(self.batch5(self.conv6(F.gelu(self.batch5(self.conv6(x)))))))

        x = self.drop(x)

        x = self.pool(F.gelu(self.batch5(self.conv6(F.gelu(self.batch5(self.conv6(x)))))))

        x = x.view(-1, 256*4*3)

        x = F.log_softmax(self.fc1(x), dim=1)

        

        return x

    

model = Net()



if train_on_gpu:

    model.cuda()
epochs = 30

lr = 0.01

criterion = nn.KLDivLoss(reduction='batchmean')

optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)

t_traj = []

v_traj = []

valid_loss_min = np.Inf



for epoch in range(epochs):

    

    train_loss = 0

    valid_loss = 0

    

#     model.train()

    for data, label in train_loader:

        if train_on_gpu:

            data = data.cuda().float()

            label = label.cuda().float()

        else:

            data = data.float()

            label = label.float()

        optimizer.zero_grad()

        output = model(data)

#         print(output.shape)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    

#     model.eval()

    with torch.no_grad():

        for data, label in valid_loader:

            if train_on_gpu:

                data = data.cuda().float()

                label = label.cuda().float()

            else:

                data = data.float()

                label = label.float()

            output = model(data)

            loss = criterion(output, label)

            valid_loss += loss.item()

    t_traj.append(train_loss/len(train_loader))

    v_traj.append(valid_loss/len(valid_loader))

    

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'best_model.pt')

        valid_loss_min = valid_loss



    print('Current Epoch: {}'.format(epoch+1))

    print('Running Loss: {:.3f}'.format(train_loss/len(train_loader)))

    print("Validation Loss: {:.3f}".format(valid_loss/len(valid_loader)))
plt.figure(figsize=(10,7))

plt.plot(list(range(1, epochs+1)), t_traj, color='dodgerblue')

plt.plot(list(range(1, epochs+1)), v_traj, color='red')

plt.title("Training Log")

plt.show()
model.load_state_dict(torch.load('best_model.pt'))
del x_data, X

del y_data, y

del dataset

del train_loader

del valid_loader

gc.collect()

gc.collect()
def test_loader(files=None, by=0):

    i = 0

    out = []

    for file in tqdm(files):

        if (100*by <= i) & (i < 100*(by+1)):

            fs, data = wavfile.read(file)

            out.append(data)    

        i+=1

    out = np.array(out)

    return out
# train_on_gpu = False



x_test = sorted(glob('/kaggle/input/dacon-sound-classification/test/*.wav'))

submission = pd.read_csv('/kaggle/input/dacon-sound-classification/submission.csv', index_col=0)



for _ in range(100):

    X = test_loader(files=x_test, by=_)



    mels = [librosa.feature.melspectrogram(y=X[i].astype(np.float), sr=16000, hop_length=128) for i in range(X.shape[0])]

    X = np.array(mels)

    X = X.reshape(-1, 1, 128, 126)

# 가장 좋은 모델의 weight를 불러옵니다.



# 예측 수행



    if train_on_gpu:

        X = torch.from_numpy(X).cuda().float()

        y = np.exp(model(X).cpu().detach().numpy())

    else:

        model.cpu()

        X = torch.from_numpy(X).float()

        y = np.exp(model(X).detach().numpy())



# 예측 결과로 제출 파일을 생성합니다.

    

    submission.iloc[100*_:100*(_+1), :] = y

    

    del X, y

    del mels

    gc.collect()

    gc.collect()

    

submission.to_csv('submission.csv')