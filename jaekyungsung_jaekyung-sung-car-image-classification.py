# Install Jovian

!pip install jovian --upgrade --quiet
import os

import torch

import torchvision

import tarfile

import torch.nn as nn

import numpy as np

from PIL import Image

import torch.nn.functional as F

from torchvision.datasets.utils import download_url

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as tt

from torch.utils.data import random_split

import matplotlib.pyplot as plt

import jovian

from scipy.io import loadmat

import pandas as pd

from torchvision import transforms

import torchvision.models as models

from skimage import io

from tqdm import tqdm

import time

from IPython.display import display



%matplotlib inline
project_title = 'car-classification'
jovian.commit(project= project_title)
# Dowload the train dataset

dataset_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"

download_url(dataset_url, '.')



# Extract from archive

with tarfile.open('./car_ims.tgz', 

                  'r:gz') as tar:

    tar.extractall(path='./data/')

    

    

    

# Download DevKit https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

devkit_dataset_url = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"

download_url(devkit_dataset_url, '.')



# Extract from archive

with tarfile.open('./car_devkit.tgz', 'r:gz') as tar:

    tar.extractall(path='./data/devkit') 



    

    

label_dataset_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"

download_url(label_dataset_url, './data')
current = os.getcwd()

print(os.listdir(current))
current = os.chdir('data')

print(os.listdir(current))
current = os.chdir('car_ims')

os.mkdir('test') # create test file 

os.mkdir('train') # create train file
os.getcwd()

os.chdir('../..')

# Look into the train directory

data_dir = './data/car_ims'



length_data = len(os.listdir(data_dir))





label_dir = './data/'

print(os.listdir(label_dir))





# Look into the devkit

dev_kit_dir = './data/devkit/devkit'



# What files are in devkit?

print(os.listdir(dev_kit_dir)) 



text_file = open('./data/devkit/devkit/README.txt')

file_content = text_file.read()

print(file_content)

text_file.close()
cars_meta = loadmat('./data/devkit/devkit/cars_meta.mat')

cars_train_annos = loadmat('./data/devkit/devkit/cars_train_annos.mat')

cars_test_annos = loadmat('./data/devkit/devkit/cars_test_annos.mat')



car_annos = loadmat(label_dir+'/cars_annos.mat')
anno_mat = loadmat('./data/cars_annos.mat')

annotations = anno_mat['annotations']
label = [c for c in car_annos['class_names'][0]]

label = pd.DataFrame(label, columns = ['Model'])

print('{} classes '.format(len(label)))

label.head(196)
labels_dict = label.to_dict()['Model']

for key, value in labels_dict.items() :

    key = int(key)

    value = str(value)
labels_dict
list_lable = list()

for example in car_annos['annotations'][0]:

    list_lable.append((example[0].item()[-10:], example[-2].item()-1, example[-1].item()))



lable_data_frame = pd.DataFrame(list_lable)

lable_data_frame.head()
tes_train_series = lable_data_frame[2]

tes_train_series.value_counts()
class_count_series = lable_data_frame[1]

class_count_series.value_counts(sort = True)



current = os.getcwd()



current = os.chdir(current + '/data/car_ims')

print(os.getcwd())
current = os.listdir(os.getcwd())

index = 0



for images in current:

    if images[-4] == '.':

        #index = df.index[df.lable_data_frame == images]

        image_test = int(lable_data_frame.iloc[index, 2])

        if image_test == 1:

            new = 'test/' + images

            os.rename(images, new)

        else:

            new = 'train/'+images

            os.rename(images, new)

        index += 1

        

        

    
os.getcwd()

os.listdir()

print(len(os.listdir('train')))

print(len(os.listdir('test')))
# Normalization and augumentation 



train_tfms = tt.Compose([tt.Resize((400, 400)),



                         tt.RandomRotation(15),

                         tt.RandomHorizontalFlip(), # only horizontal flip as vertical flip does not makes sense in this context

                         tt.ToTensor(),

                         tt.RandomErasing(inplace=True),

                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        

                         

                    

                         

                        ])



valid_tfms = tt.Compose([tt.Resize((400,400)), tt.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
class CarDatasetLoader(Dataset):

    def __init__(self, list_label, root_dir, transformation, transform = False):

        self.annotations = list_label

        self.transformation = transformation

        self.transform = transform

        self.root_dir = root_dir

        

        

    def __len__(self):

        '''return the length'''

        return len(os.listdir(self.root_dir)) # # of images 

    

    def __getitem__(self, index):

        '''Return specific item of index'''

        

        items = os.listdir(self.root_dir)



        row = self.annotations.loc[self.annotations[0] == items[index]]

        

        

        img_id = row[0].item()

        img_label = row[1].item()

        

        img_path = os.path.join(str(self.root_dir), str(img_id)) # Select the image self.annotations.iloc[index, 0]

        

        

        image = Image.open(img_path).convert('RGB')

        

        # transform 

        if self.transform:

            image = self.transformation(image)

        return image, img_label

    
test_dir = '/kaggle/working/data/car_ims/test'

train_dir = '/kaggle/working/data/car_ims/train'

test_ds = CarDatasetLoader(lable_data_frame, test_dir, transformation = valid_tfms, transform=True) # 240 by 240 pixel

train_ds = CarDatasetLoader(lable_data_frame, train_dir, transformation = train_tfms, transform = True)
print(len(test_ds))

print(len(train_ds))
batch_size = 26
# PyTorch data loaders

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True) # change num_worder to 3 

valid_dl = DataLoader(test_ds, batch_size*2, num_workers=2, pin_memory=True) # change num_worder to 3 
def show_batch(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = images

        ax.imshow(torchvision.utils.make_grid(data, nrow=8).permute(1, 2, 0))

        break
show_batch(train_dl)
show_batch(valid_dl)
def show_img(img):    

    plt.imshow(img.permute(1, 2, 0))

    plt.show()




label = labels_dict[train_ds[50][1]]



print(label)



show_img(train_ds[50][0])

label = labels_dict[train_ds[1624][1]]

print(label)

show_img(train_ds[1624][0])
label = labels_dict[train_ds[5000][1]]

print(label)

show_img(train_ds[5000][0])
def default_device():

    '''Indicate availablibity of GPU, otherwise return CPU'''

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')
def to_device(tensorss, device):

    '''Move tensor to chosen device'''

    if isinstance(tensorss, (list, tuple)):

        return [to_device(x, device) for x in tensorss]

    return tensorss.to(device, non_blocking = True)
class DeviceDataloader():

    '''Wrap DataLoader to move the model to device'''

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        '''Yield batch of data after moving the data to a device'''

        for batch in self.dl:

            yield to_device(batch, self.device)

        

    def __len__(self):

        '''Return number of batches'''

        return len(self.dl)
# Check available device type

device = default_device()

device
train_dl = DeviceDataloader(train_dl, device)

valid_dl = DeviceDataloader(valid_dl, device)
def accuracy(output, label):

    _, preds = torch.max(output, dim = 1)

    accuracy_percent = torch.tensor(torch.sum(preds == label).item() / len(preds))

    return accuracy_percent


def evaluate(model, valid_dl):

    model.eval()

    output = [model.validation_step(batch) for batch in valid_dl]

    epoch_loss = model.validation_epoch_end(output)

    return epoch_loss


def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']
def fit(epochs, model, train_loader, val_loader, scheduler,

                  opt_func=torch.optim.SGD,  weight_decay=0, grad_clip=None):

    

    torch.cuda.empty_cache()

    history = []

    model.train()

    



    

    for epoch in range(epochs):

        train_losses = []

        

        # Training Phase 

        model.train()

        

        

        for batch in tqdm(train_loader):

            loss = model.training_step(batch)

            

            

            train_losses.append(loss)

            

            

            opt_func.zero_grad()

            loss.backward()

            opt_func.step()

            

            # Gradient clipping

            if grad_clip: 

                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            

            

  

        # Validation phase

        model.eval()

    

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        

        model.epoch_end(epoch, result)

        history.append(result)

        

        

        model.train()

        scheduler.step(result['val_acc'])

    return history
class MultilabelImageClassificationBase(nn.Module):

    def training_step(self, batch):

        image, label = batch

        out = self(image) # prediction generated

        loss = F.cross_entropy(out, label) # Calculate loss using cross_entropy

        return loss

    

    def validation_step(self, batch):

        image, label = batch

        out = self(image) # predictioon generated

        loss = F.cross_entropy(out, label) # Calculate loss using cross_entropy

        acc = accuracy(out, label)

        return {'val_loss': loss.detach(), 'val_acc':acc}

    

    def validation_epoch_end(self, output):

        '''at the end of epoch, return average score (accuracy and cross entropy loss)'''

        batch_loss = [x['val_loss'] for x in output]

        epoch_loss = torch.stack(batch_loss).mean()



        batch_accs = [x['val_acc'] for x in output]

        epoch_acc = torch.stack(batch_accs).mean()  

        return {'val_loss': epoch_loss.item(), 'val_acc':epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        '''Print out the score (accuracy and cross entropy loss) at the end of the epoch'''

        # result recorded using evaluate function 

        print("Epoch [{}], train_loss: {:}, val_loss: {:}, val_acc: {:}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc'])) 
class Resnet50(MultilabelImageClassificationBase):

    def __init__(self):

        super().__init__()

        # Use a pretrained model

        self.network = models.resnet50(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Linear(num_ftrs, 196)

    

    def forward(self, xb):

        return self.network(xb)
model = to_device(Resnet50(), device)
history = evaluate(model, valid_dl)

history




epochs = 10

grad_clip = 0.1

weight_decay = 1e-4

lr = 0.0001



# Set up one-cycle learning rate scheduler

opt_func = torch.optim.Adam(model.parameters(), lr= lr)

sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_func, mode='max', patience=3)

%%time

     

history = fit(epochs = epochs, model = model, train_loader = train_dl, val_loader = valid_dl, scheduler = sched,

                  opt_func=opt_func,  weight_decay= weight_decay, grad_clip=grad_clip)
def plot_acc(history):

    scores = [x['val_acc'] for x in history]

    plt.plot(scores, '-x')

    plt.xlabel('epoch')

    plt.ylabel('score')

    plt.title('Accuracy vs. No. of epochs');
plot_acc(history)
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
def predict_image(img):

    model.eval()

    print('actual: {}'.format(labels_dict[img[1]]))

    # Convert to a batch of 1

    xb = to_device(img[0].unsqueeze(0), device)

    # Get predictions from model

    yb = model(xb)

    # Pick index with highest probability

    _, preds  = torch.max(yb.data, dim=1)

    # Retrieve the class label

    return labels_dict[preds[0].item()]
img = test_ds[50]

print('predicted: {}'.format(predict_image(img)))

show_img(img[0])
img = test_ds[150]

print('predicted: {}'.format(predict_image(img)))

show_img(img[0])
img = test_ds[250]

print('predicted: {}'.format(predict_image(img)))

show_img(img[0])
img = test_ds[523]

print('predicted: {}'.format(predict_image(img)))

show_img(img[0])
torch.save(model.state_dict(), 'car_prediction-course-project.pth')
jovian.log_metrics(train_loss=history[-1]['train_loss'], 

                   val_loss=history[-1]['val_loss'], 

                   val_acc=history[-1]['val_acc'])
jovian.log_metrics(test_loss=history[-1]['val_loss'], test_acc=history[-1]['val_acc'])
jovian.commit(project= project_title)