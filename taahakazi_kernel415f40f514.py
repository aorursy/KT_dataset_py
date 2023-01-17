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
import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import librosa.display as display

import librosa

import IPython.display as ipd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchaudio

from torch.utils.data import random_split, DataLoader, Dataset

import torchaudio.transforms as T

import torchvision

from tqdm.notebook import tqdm

import random

from scipy import signal

from scipy.io import wavfile
sample = wavfile.read('/kaggle/input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/happy/1002.wav')

sample_array = np.array(sample[1],dtype=float)

display.waveplot(sample_array, sr=sample[0])
ipd.Audio(sample_array, rate=sample[0])
sample_spec = librosa.feature.melspectrogram(sample_array, sr=16000)

a = np.max

display.specshow(librosa.core.power_to_db(sample_spec,ref= a), sr=16000,

                 x_axis='ms', y_axis='mel')

plt.show()
sample_spec.shape
data_dir = '../input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/'

classes = os.listdir(data_dir)

print(classes)

def convert():

    X = []

    for subdir, dirs, files in os.walk(data_dir + 'tree'):

        for file in files:

            x =  wavfile.read(os.path.join(subdir, file))

            x_array = np.array(x[1],dtype=float)

            X.append(x_array)

    return X
Tree = convert()

print(Tree[0])
for x in Tree:

    print(len(x))
sample_spec = librosa.feature.melspectrogram(Tree[0], sr=16000)

db_img = librosa.core.power_to_db(sample_spec,ref= a)

display.specshow(db_img, sr=16000,

                 x_axis='ms', y_axis='mel')

plt.show()

ipd.Audio(Tree[1], rate=16000)
melspec = T.MelSpectrogram(sample_rate=16000,

                                        n_fft=2048,

                                        hop_length=512)
yad = melspec(torch.Tensor(Tree[0]))

yad = yad.unsqueeze(0)

print(yad.shape)
print(Tree[0])
Tree[0].shape
print(yad)
yad.shape
print(len(Tree))
for subdir, dirs, files in os.walk(data_dir + 'tree'):

    print(len(files))
X = []

y = []

for dirname, _, filenames in os.walk('/kaggle/input/synthetic-speech-commands-dataset/augmented_dataset/augmented_dataset/'):

    melspec = T.MelSpectrogram(sample_rate=16000,n_fft=2048,hop_length=512)

    for filename in filenames:

        if dirname.split('/')[-1]:

            x = wavfile.read(os.path.join(dirname, filename))

            x_array = np.array(x[1],dtype=float)

            yad = melspec(torch.Tensor(x_array))

            yad = yad.unsqueeze(0)

            X.append(yad)

            y.append(dirname.split('/')[-1])

   
def Convert_To_Tensors(data_dir):

    melspec = T.MelSpectrogram(sample_rate=16000,n_fft=2048,hop_length=512)

    for subdir, dirs, files in os.walk(data_dir):

        for file in files:

            x =  wavfile.read(os.path.join(subdir, file))

            x_array = np.array(x[1],dtype=float)

            yad = melspec(torch.Tensor(x_array))

            yad = yad.unsqueeze(0)

            X.append(yad)

    return X
print(y)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd

mlb = MultiLabelBinarizer()



mlb.fit(pd.Series(y).fillna("missing").str.split(', '))

y_mlb = mlb.transform(pd.Series(y).fillna("missing").str.split(', '))

mlb.classes_
y_mlb = torch.tensor(y_mlb)

y_mlb_labels = torch.max(y_mlb, 1)[1]

print(y_mlb_labels)
y_mlb = torch.tensor(y_mlb_labels, dtype=torch.long)

print(y_mlb.shape)
print(X[0].shape)


X_train, X_valtest, y_train, y_valtest = train_test_split(X,y_mlb,test_size=0.2, random_state=37)

X_val, X_test, y_val, y_test = train_test_split(X_valtest,y_valtest,test_size=0.5, random_state=37)
print(len(X_train))

X_train[0][0]
print(len(X_train))

print(len(y_train))
class MyDataset(Dataset):

    def __init__(self, data, targets, transform=None):

        self.data = data

        self.targets = targets



    def __getitem__(self, index):

        x = self.data[index]

        y = self.targets[index]

            

        return x, y



    def __len__(self):

        return len(self.data)
train_ds = MyDataset(X_train,y_train)

val_ds = MyDataset(X_val,y_val)

test_ds = MyDataset(X_test,y_test)

batch_size=128

train_dl = torch.utils.data.DataLoader(train_ds, batch_size,shuffle=True, pin_memory=True,num_workers=4 )

val_dl = torch.utils.data.DataLoader(val_ds, batch_size,pin_memory=True,num_workers=4 )
print(train_ds[0][0][0][0].shape)
def accuracy(outs, labels):

    _, preds = torch.max(outs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class ModelBase(nn.Module):



    # defines mechanism when training each batch in dl

    def train_step(self, batch):

        xb, labels = batch

        outs = self(xb)

        loss = F.cross_entropy(outs, labels)

        return loss



    # similar to `train_step`, but includes acc calculation & detach

    def val_step(self, batch):

        xb, labels = batch

        outs = self(xb)

        loss = F.cross_entropy(outs, labels )

        acc = accuracy(outs,   labels)

        return {'loss': loss.detach(), 'acc': acc.detach()}



    # average out losses & accuracies from validation epoch

    def val_epoch_end(self, outputs):

        batch_loss = [x['loss'] for x in outputs]

        batch_acc = [x['acc'] for x in outputs]

        avg_loss = torch.stack(batch_loss).mean()

        avg_acc = torch.stack(batch_acc).mean()

        return {'avg_loss': avg_loss, 'avg_acc': avg_acc}



    # print all data once done

    def epoch_end(self, epoch, avgs, test=False):

        s = 'test' if test else 'val'

        print(f'Epoch #{epoch + 1}, {s}_loss:{avgs["avg_loss"]}, {s}_acc:{avgs["avg_acc"]}')
@torch.no_grad()

def evaluate(model, val_dl):

    # eval mode

    model.eval()

    outputs = [model.val_step(batch) for batch in val_dl]

    return model.val_epoch_end(outputs)





def fit(epochs, lr, model, train_dl, val_dl, opt_func=torch.optim.Adam):

    torch.cuda.empty_cache()

    history = []

    # define optimizer

    optimizer = opt_func(model.parameters(), lr)

    # for each epoch...

    for epoch in range(epochs):

        # training mode

        model.train()

        # (training) for each batch in train_dl...

        for batch in tqdm(train_dl):

            # pass thru model

            loss = model.train_step(batch)

            # perform gradient descent

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # validation

        res = evaluate(model, val_dl)

        # print everything useful

        model.epoch_end(epoch, res, test=False)

        # append to history

        history.append(res)

    return history
class Classifier(ModelBase):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(1, 512, kernel_size=3, padding=1),   # 512 x 128 x 32 

            nn.ReLU(),

            nn.BatchNorm2d(512),

            nn.MaxPool2d(2, 2),

            

            

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), # 256 x 64 x 16

            nn.ReLU(),

            nn.BatchNorm2d(256),

            nn.MaxPool2d(2, 2),



            

            nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1), # 128 x 32x 8

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.MaxPool2d(2, 2),

            

            nn.Flatten(),

            nn.Linear(8192, 64),

            nn.ReLU(),

            nn.Linear(64, 30))

        

    def forward(self, xb):

        return self.network(xb)

model = Classifier()

model
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)

device = get_default_device()

device
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device);
model = to_device(Classifier(), device)
lr = 1e-5

epochs = 10

print(val_dl)
evaluate(model, val_dl)

history= []

history += fit(epochs, lr, model, train_dl, val_dl)
plt.plot([x['avg_loss'] for x in history])

plt.title('Losses over epochs')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.show()
plt.plot([x['avg_acc'] for x in history])

plt.title('Accuracy over epochs')

plt.xlabel('epochs')

plt.ylabel('acc')

plt.show()
torch.save(model.state_dict(), 'Classifier.pth')

model.load_state_dict(torch.load('Classifier.pth'))
test_dl = torch.utils.data.DataLoader(test_ds, batch_size,pin_memory=True,num_workers=4 )

evaluate(model, test_dl)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size,pin_memory=True,num_workers=4 )

test_dl = DeviceDataLoader(test_dl, device)
evaluate(model, test_dl)
!pip install jovian --upgrade -q
import jovian
project_name = 'Audio_Classifier_1'

jovian.commit(project=project_name)