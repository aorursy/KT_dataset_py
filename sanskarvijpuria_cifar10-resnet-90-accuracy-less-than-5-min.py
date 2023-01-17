# Uncomment and run the commands below if imports fail

# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y

#!pip install matplotlib --upgrade --quiet
import os

import torch

import torchvision

import tarfile

import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from torchvision.datasets.utils import download_url

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

import torchvision.transforms as tt

from torchvision.transforms import ToTensor

from torch.utils.data import random_split

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image

import random

import pandas as pd

from tqdm import tqdm
project_name='05b-cifar10-resnet-dockship'
# Dowload the dataset

#dataset_url = "http://files.fast.ai/data/cifar10.tgz"

#download_url(dataset_url, '.')



# Extract from archive

#with tarfile.open('./cifar10.tgz', 'r:gz') as tar:

#    tar.extractall(path='./data')

    

# Look into the data directory

data_dir = '../input/cifar10-dockship/CIFAR10'

print(os.listdir(data_dir))

classes = os.listdir(data_dir + "/train")

print(classes)
# Data transforms (normalization & data augmentation)

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 

                         tt.RandomHorizontalFlip(), 

                         tt.ToTensor(), 

                         tt.Normalize(*stats,inplace=True)])

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
# PyTorch datasets

dataset = ImageFolder(data_dir+'/train/', train_tfms)

#valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

dataset
batch_size = 400
random_seed = 21

torch.manual_seed(random_seed);

from torch.utils.data import random_split

val_size = 5000

train_size = len(dataset) - val_size



train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
# PyTorch data loaders

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

valid_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)
def show_batch(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))

        break
show_batch(train_dl)
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

valid_dl = DeviceDataLoader(valid_dl, device)
class SimpleResidualBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu2 = nn.ReLU()

        

    def forward(self, x):

        out = self.conv1(x)

        out = self.relu1(out)

        out = self.conv2(out)

        return self.relu2(out) + x # ReLU can be applied before or after adding the input
simple_resnet = to_device(SimpleResidualBlock(), device)



for images, labels in train_dl:

    out = simple_resnet(images)

    print(out.shape)

    break

    

del simple_resnet, images, labels

torch.cuda.empty_cache()
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

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

        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
from torchvision import models
def conv_block(in_channels, out_channels, pool=False):

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 

              nn.BatchNorm2d(out_channels), 

              nn.ReLU(inplace=True)]

    if pool: layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)



class ResNet9(ImageClassificationBase):

    ''' def __init__(self, in_channels, num_classes):

        super().__init__()

        

        self.conv1 = conv_block(in_channels, 64)

        self.conv2 = conv_block(64, 128, pool=True)

        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        

        self.conv3 = conv_block(128, 256, pool=True)

        self.conv4 = conv_block(256, 512, pool=True)

        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        

        self.classifier = nn.Sequential(nn.MaxPool2d(4), 

                                        nn.Flatten(), 

                                        nn.Linear(512, num_classes))

        

    def forward(self, xb):

        out = self.conv1(xb)

        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.res2(out) + out

        out = self.classifier(out)

        return out'''

    def __init__(self):

        super().__init__()

        # Use a pretrained model

        

        self.network = models.vgg19(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.classifier[6].in_features

        self.network.classifier[6] = nn.Linear(num_ftrs, 10)

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))

    

    def freeze(self):

        # To freeze the residual layers

        for param in self.network.parameters():

            param.require_grad = False

        for param in self.network.classifier[6].parameters():

            param.require_grad = False

    

    def unfreeze(self):

        # Unfreeze all layers

        for param in self.network.parameters():

            param.require_grad = True
model = ResNet9()
model = to_device(model, device)

model
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 

                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):

    torch.cuda.empty_cache()

    history = []

    

    # Set up cutom optimizer with weight decay

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 

                                                steps_per_epoch=len(train_loader))

    

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        lrs = []

        for batch in tqdm(train_loader):

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            

            # Gradient clipping

            if grad_clip: 

                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            

            optimizer.step()

            optimizer.zero_grad()

            

            # Record & update learning rate

            lrs.append(get_lr(optimizer))

            sched.step()

        

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['lrs'] = lrs

        model.epoch_end(epoch, result)

        history.append(result)

    return history
history = [evaluate(model, valid_dl)]

history
epochs = 15

max_lr = 0.01

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)
model.freeze()
epochs = 25

max_lr = 0.001

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)
model.unfreeze()
epochs = 25

max_lr = 1e-4

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)
train_time='4:07'
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');
plot_accuracies(history)
def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]

    val_losses = [x['val_loss'] for x in history]

    plt.plot(train_losses, '-bx')

    plt.plot(val_losses, '-rx')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');
plot_losses(history)
def plot_lrs(history):

    lrs = np.concatenate([x.get('lrs', []) for x in history])

    plt.plot(lrs)

    plt.xlabel('Batch no.')

    plt.ylabel('Learning rate')

    plt.title('Learning Rate vs. Batch no.');
plot_lrs(history)
#!wget "https://www.kaggleusercontent.com/kf/41672076/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..tRJKi16iMS1QMl93Xddirw.TPI5EMlCne9y_W-kfcx8GkjUFRItNQE8cEj4EgKmSw2idADv8IqAHFK3pj9XJxLNC63xBCtxGigX66xkWfpk9b3t3plBA9CPxLTSts_m8pdxH7Zo22n0ZXrpy2zIlbzN7syAJ-UTlAzFjJdQxMpOlTUTBBhi7CBA6bj0Ga-8_qxHSXCPjd1s8TkgGmWRjIZTMlo9WprsK1RtHxq9Gu6BY3-R6dkMh_NmCuKV2QUgL6NTe0PLtnNPSOf9tWuFx2QBaDYlljrd1OO7hRdVDOkHT5qo05zDPAlSNY5hqOIKzT18bdIwh6L7Y61NGnJpLHtkcSCZjIzxPfLaX63bR4tlpQLeo7L2WMnNRnSqb8EthcCfF9GghV2cTuievjKdBsEi9P9Nhkfh2wrOoxI7QTpS38SPyY_WJX4fO4QNsiOmsD3HmIYLFuH0xgHHi49ykw4Cy_TK0EbSPQczmYGZ9L3N2k2tsr7Lvrt3ysnFxFxWFGGrhPygNT5Jy1sRdDnTnL4It9wWP8UG3WD9AA7L3eFr1iuJGO-njXToqqyhF2o3UwHF1BnFT7nAI86vVR2Lh2-211rwNM2Rm7PXZxJ7FdyQ4BoEOUb8ZfxDfake0clRWbq_uYTcW7EwpYvGyuhjH0yXmnmJlEsWRDv3847X1RnFF1y8yGZp-PbBVmwnXeGzW_M.eIeUyaAYij6tXgT6_uWQ6Q/cifar10-resnet9.pth"
#model.load_state_dict(torch.load('./cifar10-resnet9.pth'))
import glob

class CustomDatasetFromFile(torch.utils.data.Dataset):

    def __init__(self, folder_path,transform=ToTensor()):

        """

        A dataset example where the class is embedded in the file names

        This data example also does not use any torch transforms



        Args:

            folder_path (string): path to image folder

        """

        # Get image list

        self.image_list = glob.glob(folder_path+'*')

        # Calculate len

        self.data_len = len(self.image_list)

        self.transform=transform



    





    def __getitem__(self, index):

        # Get image name from the pandas df

        single_image_path = self.image_list[index]

        # Open image

        im_as_im = Image.open(single_image_path).convert("RGB")

        

        # Do some operations on image

        # Convert to numpy, dim = 28x28

        #im_as_np = np.asarray(im_as_im)/255

        # Add channel dimension, dim = 1x28x28

        # Note: You do not need to do this if you are reading RGB images

        # or i there is already channel dimension

        #im_as_np = np.expand_dims(im_as_np, 0)

        # Some preprocessing operations on numpy array

        # ...

        # ...

        # ...



        # Transform image to tensor, change data type

        #im_as_ten = torch.from_numpy(im_as_np).float()

        im_as_ten = self.transform(im_as_im).float()



        # Get label(class) of the image based on the file name

        class_indicator_location = single_image_path.rfind('g')

        label = single_image_path[class_indicator_location-13:class_indicator_location+1]

        #label=[ord(c) for c in label]



        #print(label)

        #label=float(label)

        #label= torch.tensor(label, dtype=torch.long)

        return (im_as_ten, label)



    def __len__(self):

        return self.data_len
testset= CustomDatasetFromFile(data_dir+'/test/')

testset.__getitem__(90)
def decode_target(target,imagefname, labels=classes, threshold=0.8):

    

    #print(type(target))

    if torch.is_tensor(target)==True:

        max_tensor,index, = torch.topk(target,1)

        index=int(index)

        #print(index)

        #result.append((imagefname,labels[index]))

        return (imagefname,labels[index])

    #below method in else is for single image which does not have the TARGET in tensor form. The target is in int form.

    elif (isinstance(target, int))==True:

        #result.append(imagefname,labels[target])

        return (imagefname,labels[target])

    else:

        print("Wrong label type")
def show_sample(img, target,imagefname, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

        

    result=decode_target(target,imagefname)

    print('Labels:',result[1] )

    print('Filename:',result[0])
def predict_single(image,imagefname):

    xb = image

    xb=xb.unsqueeze(0) 

    #xb= torch.randn(1, 3, 32, 32)

    xb = to_device(xb, device)

    #print(xb.size())

    #preds = final_model(xb)

    #preds = model(xb)

    preds = model(xb)

    prediction = preds[0]

    #print("Prediction: ", prediction)

    #print(type(prediction))

    #show_sample(image, prediction, imagefname)

    result=decode_target(prediction,imagefname)

    

    #print('Labels:',result[1] )

    #print('Filename:',result[0])

    return result
filename=[]

label=[]

count=0

for image in testset:

  #print(image)

  filen, lb=predict_single(*image)

  filename.append(filen)

  label.append(lb)
print(filename[0:100])

print(label[0:100])
df = pd.DataFrame()

df["filename"]=filename

df["label"]=label

df.sort_values(by=['filename'], inplace=True)

df
img_array = np.array(Image.open(data_dir+'/test/00003_test.png'))

plt.imshow(img_array)
df.to_csv('submission2.csv',index=False)
torch.save(model.state_dict(), 'cifar10-resnet9.pth')
!pip install jovian --upgrade --quiet
import jovian
jovian.reset()

jovian.log_hyperparams(arch='resnet9', 

                       epochs=epochs, 

                       lr=max_lr, 

                       scheduler='one-cycle', 

                       weight_decay=weight_decay, 

                       grad_clip=grad_clip,

                       opt=opt_func.__name__)
jovian.log_metrics(val_loss=history[-1]['val_loss'], 

                   val_acc=history[-1]['val_acc'],

                   train_loss=history[-1]['train_loss'],

                   time=train_time)
jovian.commit(project=project_name, environment=None, outputs=['cifar10-resnet9.pth'])