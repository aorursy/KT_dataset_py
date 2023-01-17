import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import specgram

import pandas as pd

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys
RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

dir_list = os.listdir(RAV)

dir_list.sort()



emotion = []

gender = []

path = []

for i in dir_list:

    fname = os.listdir(RAV + i)

    for f in fname:

        part = f.split('.')[0].split('-')

        emotion.append(int(part[2]))

        temp = int(part[6])

        path.append(RAV + i + '/' + f)



        

RAV_df = pd.DataFrame(emotion)



RAV_df = RAV_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})

RAV_df.columns = ['emotion']

RAV_df['labels'] = RAV_df.emotion 

RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)

RAV_df = RAV_df.drop(['emotion'], axis=1)

RAV_df.labels.value_counts()

# fear

fname = RAV + 'Actor_14/03-01-06-02-02-02-14.wav'  

data, sampling_rate = librosa.load(fname, sr=None)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



ipd.Audio(fname)
# happy

fname = RAV + 'Actor_14/03-01-03-02-02-02-14.wav'  

data, sampling_rate = librosa.load(fname, sr = None)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



ipd.Audio(fname)
# Source - RAVDESS; Gender - Male; Emotion - Angry 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"

X, sample_rate = librosa.load(path, sr=None)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=16)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title(f'Audio sampled at {sample_rate} hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
import numpy as np 

import pandas as pd 

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from torch.utils.data import Subset, DataLoader

import librosa

import os

import random







def seed_everything(seed):

    """Function to enable reproducibility"""

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True





class RAVDataset(Dataset):

    """Dataset for wav files from RAVDESS with their mel frequency

    cepstrum coefs"""

    def __init__(self, data_root):

        self.samples = []

        for actor in sorted(os.listdir(data_root)):

            actor_folder = os.path.join(data_root, actor)

            

            for actor_wav in sorted(os.listdir(actor_folder)):

                codes = actor_wav.split('.')[0].split('-')

                emotion = int(codes[2]) - 1

                intensity = codes[3]

                actor_id = codes[6]

                wav_path = os.path.join(actor_folder, actor_wav)

                # create image

                wav, sample_rate = librosa.load(wav_path, sr=None,duration=2,offset=0.75)

                mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=32)

                self.samples.append((torch.from_numpy(mfcc[np.newaxis,...]), emotion))

                

    def __len__(self):

        return len(self.samples)

    

    def __getitem__(self, idx):

        return self.samples[idx]

    

def train_val_dataset(dataset, val_split=0.20, test_split = 0.10):

    """Get train, val, test datasets"""

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split+test_split)

    datasets = {}

    split = int((test_split/(val_split+test_split))*len(val_idx))

    test_idx = val_idx[:split]

    val_idx = val_idx[split:]

    #print(len(train_idx),len(val_idx),len(test_idx))

    datasets['train'] = Subset(dataset, train_idx)

    datasets['val'] = Subset(dataset, val_idx)

    datasets['test'] = Subset(dataset, test_idx)

    return datasets
SEED = 42

seed_everything(SEED)

RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

data = RAVDataset(RAV)

datasets = train_val_dataset(data)

dataloaders = {x:DataLoader(datasets[x],8, shuffle=True) for x in ['train','val','test']}
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=13, stride=1, padding=6)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5)

        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=9, stride=1, padding=4)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)

        self.conv4_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(0.2)

        self.drop2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(32*2*11, 400)

        self.dense1_bn = nn.BatchNorm1d(400)

        self.fc2 = nn.Linear(400, 8)

        

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))

        x = self.pool(F.relu(self.conv3(x)))

        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))





        x = x.view(-1, 32 * 2 * 11)

        

        # classifier

        x = self.drop1(x)

        x = F.relu(self.dense1_bn(self.fc1(x)))

        x = self.drop2(x)

        x = self.fc2(x)

        return x
model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def train(epoch):

    model.train()

    tr_loss = 0

    correct = 0

    total = 0

    train_loader = dataloaders['train']

    for batch_idx, (data, target) in enumerate(train_loader):

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()



        optimizer.zero_grad()

        output = model(data)

        pred = torch.max(output.data, 1)[1]

        correct += (pred == target).sum()

        total += len(data)

        

        loss = criterion(output, target)

        # print(loss)

        

        loss.backward()

        optimizer.step()

        

        

        tr_loss = loss.item()

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {} %'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.item(),100 * correct / total))

            torch.save(model.state_dict(), './model.pth')

            torch.save(model.state_dict(), './optimizer.pth')

    train_loss.append(tr_loss / len(train_loader))

    train_accuracy.append(100 * correct / total)
def evaluate(data_loader):

    model.eval()

    loss = 0

    correct = 0

    total = 0

    for data, target in data_loader:

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = torch.max(output.data, 1)[1]

        total += len(data)

        correct += (pred == target).sum()

    loss /= len(data_loader.dataset)

    valid_loss.append(loss)    

    valid_accuracy.append(100 * correct / total)

    print('\nAverage Validation loss: {:.5f}\tAccuracy: {} %'.format(loss, 100 * correct / total))
n_epochs = 200

train_loss = []

train_accuracy = []

valid_loss = []

valid_accuracy = []

for epoch in range(n_epochs):

    train(epoch)

    evaluate(dataloaders['val'])
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

def plot_graph(epochs):

    fig = plt.figure(figsize=(20,4))

    ax = fig.add_subplot(1, 2, 1)

    plt.title("Train - Validation Loss")

    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')

    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('loss', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')

    

    ax = fig.add_subplot(1, 2, 2)

    plt.title("Train - Validation Accuracy")

    plt.plot(list(np.arange(epochs) + 1) , train_accuracy, label='train')

    plt.plot(list(np.arange(epochs) + 1), valid_accuracy, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('accuracy', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')



plot_graph(n_epochs)
test_loader = dataloaders['test']
dataiter = iter(test_loader)

images, labels = dataiter.next()

classes = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearfull', 'disgust', 'surprised')

print('GroundTruth: ', ' '.join('%9s' % classes[labels[j]] for j in range(8)))

outputs = model(images.cuda())



_, predicted = torch.max(outputs, 1)

print('Predicted   : ', ' '.join('%9s' % classes[predicted[j]]

                              for j in range(8)))
correct = 0

total = 0

with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].cuda(), data[1].cuda()

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 144 test images: %d %%' % (

    100 * correct / total))
class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))

with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].cuda(), data[1].cuda()

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(8):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(8):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))
import seaborn as sns



def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

            confusion_matrix, index=classes, columns=classes, 

        )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True)

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")



    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



nb_classes = 8



confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():

    for i, (inputs, pclasses) in enumerate(dataloaders['test']):

        inputs = inputs.cuda()

        pclasses = pclasses.cuda()

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        for t, p in zip(pclasses.view(-1), preds.view(-1)):

                confusion_matrix[t.long(), p.long()] += 1



print_confusion_matrix(confusion_matrix.numpy(), class_names = classes)