import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T

import random
import math

from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

%matplotlib inline
np.random.seed(43)
PROJECT_NAME='final-course-project'
!pip install jovian --upgrade -q
import jovian
dir(jovian)
jovian.commit(project=PROJECT_NAME, environment=None)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

bad_file = False

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):  # If it does not end with .jpg or .jpeg or .png extension
            bad_file = True
            print(os.path.join(dirname, filename))  # Show the file that does not have a .jpg or .jpeg extension
            print('-'*80)  # Print a line just under the file that does not end with .jpg or .jpeg or .png extension

if bad_file == False:
    print("All files in the dataset are of .jpg or .jpeg or .png format only.")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_DIR = os.path.join('..', 'input', 'newFlowers')
print(DATA_DIR)
print(type(DATA_DIR))
os.listdir(DATA_DIR)
os.getcwd()
class DataLoadingPipeline:
    def __init__(self, data_dir: str, validation_fraction:float=0.15,
                 test_fraction:float=0.2, train_transforms:list=[T.ToTensor(),],
                 valid_and_test_transforms:list=[T.ToTensor(),], batch_size:int=64):
        """
            This function sets up the basic data loader for the dataset. The data loaded is not moved to GPU by this method.
            data_dir: {str}  ==> This is the absolute path to the dataset folder
            validation_fraction: {float} ==> This is the fraction of train set that will be used for validation
            test_fraction: {float} ==> This is the fraction of dataset that will be used for testing, rest will be used for training (and validation of course)
            train_transforms: {list} ==> This is the list of transforms which will be applied to the train set
            valid_and_test_transforms: {list} ==> This is the list of transforms which will be applied to the validation and test set
        """
        # Creating the transforms pipeline for each of the fractions of the dataset
        self.train_transforms = T.Compose(train_transforms)
        self.valid_transforms = T.Compose(valid_and_test_transforms)
        self.test_transforms  = T.Compose(valid_and_test_transforms)
        
        # Reading the folder containing the images for creating the initial train, validation and test datasets
        self.train_data = ImageFolder(data_dir, transform=self.train_transforms)
        self.validation_data = ImageFolder(data_dir, transform=self.valid_transforms)
        self.test_data = ImageFolder(data_dir, transform=self.test_transforms)
        self.train_validation_data = ImageFolder(data_dir, transform=self.train_transforms)
        
        # Creating a dictionary for storing the conversion from the lable number to the flower name
        self.classes = {}
        # Print out the classes in the dataset along with their corresponding index
        print("The classes in this dataset are:")
        for ctr, i in enumerate(self.train_data.classes):
            print(f"{ctr}: {i.capitalize()}")
            self.classes[ctr] = i
        print()  # Print a newline for better output formatting

        self.count = num_train = len(self.train_data)     # Get the total number of images in the dataset
        print(f"The dataset has {num_train} images")
        indices = list(range(num_train))  # Create a list of indices for the all images in the dataset

        test_split = int(np.floor(test_fraction * num_train))  # Getting the number of images in the test set
        train_validation_split = num_train - test_split        # Getting the number of images in the train and validation set
        validation_split = int(np.floor(validation_fraction * train_validation_split))  # Getting the number of images in the validation set
        train_split = train_validation_split - validation_split  # Getting the number of images in the train set

        # Construct a new Generator with the default BitGenerator (PCG64), this will hellp us shuffle the indices randomly
        rng = np.random.default_rng()
        rng.shuffle(indices)  # Shuffling the indices so that every set gets approximately equal number of images for each class

        # Splitting the indices list into train_validation and test indices lists
        train_validation_idx, test_idx = indices[test_split:], indices[:test_split]
        
        # Reshuffling the train_validation indices list; preparing it for another split
        rng = np.random.default_rng()
        rng.shuffle(train_validation_idx)
        
        # Further split the train_validation indices list into train indices list and validation indices list
        train_idx, validation_idx = train_validation_idx[validation_split:], train_validation_idx[:validation_split]

        # Just for a sanity check, lets check if the train, validation and test sets have all got unique indices and none have overlapped
        # as that would mean that the corresponding image is in both the sets its indice occurs in
        if not set(train_idx).intersection(set(validation_idx)) and not set(validation_idx).intersection(set(test_idx)) and \
           not set(train_idx).intersection(set(test_idx)):
            print("[PASS] The splits are mutually exculsive of each other!")
        else:
            print("[FAIL] The splits are not mutually exculsive of each other!")
            
        
        
        # We now create random samplers to take random samples of indices from the indices lists we created.
        # This will make sure that the train set only accesses the images referred to by the train images indices list
        # and the same applies to the validation and test sets and their validation and test images indices lists.
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        # This will be used to make sure that the model gets an opportunity to train on more data.
        # As we cannot use the test set for this (for obvious reasons), we instead use the validation set, so hence the below dataloader
        # for using the train and validation set together for training, this will be used towards the end of the training process
        train_validation_sampler = SubsetRandomSampler(train_idx + validation_idx)  # This is a combination of the train and validation sets
        
        # We now create the dataloaders for the train, validation and test sets
        self.train_loader = DataLoader(self.train_data, sampler=train_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.validation_loader = DataLoader(self.validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, sampler=test_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.train_validation_loader = DataLoader(self.train_validation_data, sampler=train_validation_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
       
RESIZE_DIM = 300
IMG_DIM = 256     # We want all images to be of dimension 128x128
BATCH_SIZE = 128  # 64
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


TRAIN_TRANSFORMS = [
                    T.Resize(RESIZE_DIM, interpolation=Image.BICUBIC),
                    T.CenterCrop(IMG_DIM),
                    T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
                    T.RandomHorizontalFlip(), 
                    # T.RandomCrop(IMG_DIM, padding=8, padding_mode='reflect'),
                    # T.RandomRotation(10),  #  Did not give any improvements, some images lost important details as they go cut off
                    T.ToTensor(), 
                    # T.Normalize(*imagenet_stats,inplace=True),  #  Did not give good results, converted some images into just white squares
                    T.RandomErasing(inplace=True, scale=(0.01, 0.23)),
                   ]

VALIDATION_and_TEST_TRANSFORMS = [
                                  T.Resize(RESIZE_DIM, interpolation=Image.BICUBIC),
                                  T.CenterCrop(IMG_DIM),
                                  T.ToTensor(), 
                                  # T.Normalize(*imagenet_stats)  #  Did not give good results, converted some images into just white squares
                                 ]
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.2


flowers_data_loader = DataLoadingPipeline(
                                          data_dir=DATA_DIR,
                                          validation_fraction=VALIDATION_FRACTION,
                                          test_fraction=TEST_FRACTION,
                                          train_transforms=TRAIN_TRANSFORMS,
                                          valid_and_test_transforms=VALIDATION_and_TEST_TRANSFORMS,
                                          batch_size=BATCH_SIZE
                                         )
help(jovian.log_dataset)
train_loader_batches_count, test_loader_batches_count, validation_loader_batches_count = len(flowers_data_loader.train_loader),\
                                                                                         len(flowers_data_loader.test_loader),\
                                                                                         len(flowers_data_loader.validation_loader)

# Let's see the batch sizes for the different sets of data
print(f"{'The number of training batches are': <40} {train_loader_batches_count:^4}, each of size of {BATCH_SIZE: ^4}")
print(f"{'The number of testing batches are': <40} {test_loader_batches_count: ^4}, each of size of {BATCH_SIZE: ^4}")
print(f"{'The number of validation batches are': <40} {validation_loader_batches_count: ^4}, each of size of {BATCH_SIZE: ^4}")

# Also lets show the data in the form of a simple tuple representation
print(f"({train_loader_batches_count}, {test_loader_batches_count}, {validation_loader_batches_count})")
jovian.log_dataset(dataset_url='https://www.kaggle.com/alxmamaev/flowers-recognition',
                   val_fraction=VALIDATION_FRACTION,
                   test_fraction=TEST_FRACTION,
                   train_batches=train_loader_batches_count,
                   test_batches=test_loader_batches_count,
                   validation_batches=validation_loader_batches_count)
flowers_data_loader.train_loader.dataset.classes == flowers_data_loader.test_loader.dataset.classes
flowers_data_loader.test_loader.dataset.classes == flowers_data_loader.validation_loader.dataset.classes
for images, labels in flowers_data_loader.train_loader:
    print(images)
    print(labels)
    break
for images, labels in flowers_data_loader.train_validation_loader:
    print(images)
    print(labels)
    break
for images, labels in flowers_data_loader.test_loader:
    print(images)
    print(labels)
    break
def show_sample(data_item_obj, classes:dict, invert:bool=False):
    print("The tensor representing the image and the target image", data_item_obj)
    img, target = data_item_obj  # This is a particular data item from the data set having its own image and label
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    plt.title(classes[target])
    print('Labels:', classes[target])
def random_no_gen(no_of_elements:int, lower_limit:int=0) -> int:
    return lower_limit + math.floor(random.random() * no_of_elements)
show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)
show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)
show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)
show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)
show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)
def show_batch(dl, invert:bool=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(8, 16))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=8).permute(1, 2, 0))
        break
show_batch(flowers_data_loader.train_loader, invert=True)
show_batch(flowers_data_loader.train_loader)
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
flowers_data_loader.train_loader = DeviceDataLoader(flowers_data_loader.train_loader, device)
flowers_data_loader.validation_loader = DeviceDataLoader(flowers_data_loader.validation_loader, device)
flowers_data_loader.test_loader = DeviceDataLoader(flowers_data_loader.test_loader, device)
flowers_data_loader.train_validation_loader = DeviceDataLoader(flowers_data_loader.train_validation_loader, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    @torch.no_grad()
    def evaluate_test_set(self, test_dataset):
        result = evaluate(self, test_dataset)
        print("The results are: test_loss: {:.4f}, test_acc: {:.4f}".format(result['val_loss'], result['val_acc']))
        return {'test_loss': result['val_loss'], 'test_acc': result['val_acc']}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
COMMON_IN = 2048  # The input to the classifier layer is 2048 from the rest of ResNet50, so this value is fixed for ResNet50
NUM_CLASSES = 5   # There are 5 classes of flowers and for each we need to return a prediction probability, in the end

CLASSIFIER_ARCHITECTURES = {
                            "Simple With Dropout": nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES)),
                            "Simple Without Dropout": nn.Sequential(nn.Linear(2048, 128),nn.ReLU(), nn.Linear(128, NUM_CLASSES)),
                            "Medium With Dropout": nn.Sequential(nn.Linear(2048, 256), nn.Dropout(0.1), nn.ReLU(), nn.Linear(256, 64), nn.Dropout(0.01), nn.ReLU(), nn.Linear(64, NUM_CLASSES)),
                            "Medium Without Dropout": nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES))
                            }

class ResNet50(ImageClassificationBase):
    def __init__(self, num_classes:int):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = CLASSIFIER_ARCHITECTURES["Medium With Dropout"]
        
        for param in self.model.fc.parameters():
            param.require_grad = True
        
    def forward(self, xb):
        return self.model(xb)

    def switch_on_gradients(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def switch_off_gradients_except_classifier(self):
        # We first switch off the requires_grad parameter for all the layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # We then only switch on the requires_grad parameter for the layers of the (fc) classifer layer
        for param in self.model.fc.parameters():
            param.require_grad = True
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
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader),
                                                cycle_momentum=True)
    
    for epoch in range(epochs):
        # Training Phase 
        model.train() # Switches on training mode
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad() # reset the gradients
            
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
model = to_device(ResNet50(num_classes=5), device)
model
jovian.commit(project=PROJECT_NAME, environment=None)
torch.cuda.empty_cache()
lrs = []
epochs_list = []
train_times = []
history = [evaluate(model, flowers_data_loader.validation_loader)]
history
epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-5
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)
%%time
history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_loader, flowers_data_loader.validation_loader, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
train_times.append('5min 36s')
model.switch_on_gradients()
model
epochs = 20
max_lr = 0.001
grad_clip = 0.05
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)
%%time
history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_loader, flowers_data_loader.validation_loader, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
train_times.append('4min 4s')
def save_clear_reload_model(model):
    torch.save(model.state_dict(), 'flowers_cnn.pth')
    torch.cuda.empty_cache()
    model = to_device(ResNet50(num_classes=5), device)
    model.load_state_dict(torch.load('flowers_cnn.pth'))
save_clear_reload_model(model)
model.switch_on_gradients()
model
epochs = 10
max_lr = 0.001
grad_clip = 0.015
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)
%%time
history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_validation_loader,flowers_data_loader.test_loader, 
                                           grad_clip=grad_clip, 
                                           weight_decay=weight_decay, 
                                           opt_func=opt_func)
train_times.append('2min 19s')
save_clear_reload_model(model)
epochs = 10
max_lr = 0.0001
grad_clip = 0.005
weight_decay = 1e-5
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)
%%time
history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_validation_loader, flowers_data_loader.test_loader,
                                           grad_clip=grad_clip, 
                                           weight_decay=weight_decay, 
                                           opt_func=opt_func)
train_times.append('2min 20s')
T_MODEL = "RESNET-50-Pretrained"
CLASSIFER_LAYER_1 = 2048
CLASSIFER_LAYER_2 = 256
CLASSIFER_LAYER_3 = 64
CLASSIFER_LAYER_4 = 5
jovian.log_hyperparams(arch=f"{T_MODEL} --> Classifer layers: ({CLASSIFER_LAYER_1}, {CLASSIFER_LAYER_2}, {CLASSIFER_LAYER_3}, {CLASSIFER_LAYER_4})", 
                       lrs=lrs, 
                       epochs=epochs_list,
                       times=train_times,
                       img_dimensions=IMG_DIM,
                       batch_size=BATCH_SIZE,
                       validation_fraction=VALIDATION_FRACTION,
                       test_fraction=TEST_FRACTION
                       )
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
history.append(model.evaluate_test_set(flowers_data_loader.test_loader))
history[-1]
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    print("The prediction tensor is:")
    for i in range(len(flowers_data_loader.classes)):
        print(f"{flowers_data_loader.classes[i]:^10}"," : ",F.softmax(yb, dim=1)[0][i].item())
    # Retrieve the class label
    return flowers_data_loader.classes[preds[0].item()]
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break
other_imgs_path = os.path.join('..', 'input', 'other_imgs')
other_imgs_path
other_images = ImageFolder(other_imgs_path, transform=T.Compose(VALIDATION_and_TEST_TRANSFORMS))
LEN_OF_OTHER_IMAGES = len(other_images)
LEN_OF_OTHER_IMAGES
other_images.classes
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    print("The prediction tensor is:")
    for i in range(len(flowers_data_loader.classes)):
        print(f"{flowers_data_loader.classes[i]:^10}"," : ",F.softmax(yb, dim=1)[0][i].item())
    # Retrieve the class label
    return flowers_data_loader.classes[preds[0].item()]
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', other_images.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))
jovian.log_metrics(test_loss=history[-1]['test_loss'],
                   test_acc=history[-1]['test_acc'])
jovian.commit(project=PROJECT_NAME, environment=None)
torch.save(model.state_dict(), 'flowers_cnn.pth')
sanity_check_model = to_device(ResNet50(num_classes=5), device)
sanity_check_model.load_state_dict(torch.load('flowers_cnn.pth'))
sanity_check_model.evaluate_test_set(flowers_data_loader.test_loader)
jovian.commit(project=PROJECT_NAME, outputs=['flowers_cnn.pth'], environment=None)