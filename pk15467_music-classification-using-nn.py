# Project name used for jovian.commit
project_name = 'music-classification-using-deep-learning-with-pytorch'
# Uncomment and run the commands below if imports fail
# !conda install numpy pytorch torchaudio cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
# !conda install -c conda-forge librosa
!pip install jovian --upgrade --quiet
# !mkdir genres && wget http://opihi.cs.uvic.ca/sound/genres.tar.gz  && tar -xf genres.tar.gz genres/
# !rmdir genres
# !rm genres.tar.gz
# !rm -rf img_data
import jovian
import os
import pathlib
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor,transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.autonotebook import tqdm
from skimage.io import imread, imsave
%matplotlib inline
data_path = '../input/genres-data-for-music-classification/genres'
img_path = 'img_data'
cmap = plt.get_cmap('inferno') # this is for img color
plt.figure(figsize=(8,8)) # img size
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split() # all possible  music class
for g in genres:
    pathlib.Path(f'{img_path}/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'{data_path}/{g}'):
        songname = f'{data_path}/{g}/{filename}'
#         print(songname)
#         break
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'{img_path}/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
audio_data = data_path+'/classical/classical.00009.wav'
x , sr = librosa.load(audio_data)
print(type(x), type(sr))
#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
import IPython.display as ipd
ipd.Audio(audio_data)
import matplotlib.image as mpimg
img=mpimg.imread(img_path+'/blues/blues00093.png')
imgplot = plt.imshow(img)
plt.show()
print('shape of image is:',img.shape)
#parameters
batch_size = 32
im_size = 576
def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data,_ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()

train_transforms = transforms.Compose([transforms.Resize((im_size,im_size)),transforms.ToTensor()])
train_data = torchvision.datasets.ImageFolder(root = img_path, transform = train_transforms)
train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)
mean,std = normalization_parameter(train_loader)

train_transforms = transforms.Compose([transforms.Resize((im_size,im_size)),
                                        transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
                                        transforms.RandomRotation(degrees=10),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(size=299),  # Image net standards
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
test_transforms = transforms.Compose([
                                        transforms.Resize((im_size,im_size)),
                                        transforms.CenterCrop(size=299),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
])
#data loader
dataset = torchvision.datasets.ImageFolder(root = img_path, transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = img_path, transform =  test_transforms)
#encoder and decoder to convert classes into integer
def encoder(data):
    #label of classes
    classes = data.classes
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]
    return encoder

def decoder(data):
    #label of classes
    classes = data.classes
    
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    return decoder
#plotting rondom images from dataset
def class_plot(data,n_figures = 4):
    n_row = int(n_figures/4)
    fig,axes = plt.subplots(figsize=(24, 20), nrows = n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0,len(data))
        (image,label) = data[a]
#         print(type(image))
        label = int(label)
        encoders = encoder(data)
        l = encoders[label]
        image = image.numpy().transpose(1,2,0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()

class_plot(dataset)
class_plot(train_data)
torch.manual_seed(43)
val_size = int(len(dataset)*0.2)
train_size = len(dataset) - val_size
from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [train_size,val_size])
len(train_ds), len(val_ds)
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
def accuracy(outputs,labels):
    _,preds =torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
arch = "5 layer ( 3*299*299,1024,512,128,32,10)"
class ClassifyMusic(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,1024)
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,128)
        self.linear4 = nn.Linear(128,32)
        self.linear5 = nn.Linear(32,output_size)
    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out
    def training_step(self,batch):
        image,labels =batch
        out = self(image)
        loss =F.cross_entropy(out,labels)
        return loss
   
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

        

def evaluate(model,val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_loader,val_loader,opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
device
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
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
input_size = 3*299*299
output_size = 10
input_size
model = to_device(ClassifyMusic(input_size,output_size), device)
history = [evaluate(model, val_loader)]
history
# epochs = {0.1:50,0.01:50,0.001:50,0.0001:50}
epochs = {1e-2:50,1e-2:50,1e-3:50,1e-5:50,1e-6:50}
for lr,epoch in epochs.items():
    print(f'epoch:{epoch},lr:{lr}')
    history += fit(epoch,lr, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
test = evaluate(model, test_loader)
test_acc = test['val_acc']
test_loss = test['val_loss']
test_loss,test_acc
torch.save(model.state_dict(), project_name+'.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch,lrs=list(epochs.keys()),epochs=list(epochs.values()))
jovian.log_metrics(test_loss=test['val_loss'], test_acc=test['val_acc'])
jovian.commit(project=project_name,output=[project_name+'.pth'], environment=None)
