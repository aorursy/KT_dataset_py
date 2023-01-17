# Uncomment and run the commands below if imports fail
# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Normalize, RandomPerspective, RandomErasing, RandomGrayscale, ColorJitter
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
# Project name used for jovian.commit
project_name = '03-cifar10-feedforward'
dataset = CIFAR10(root='data/', download=True, transform=torchvision.transforms.Compose([ToTensor()]))
test_dataset = CIFAR10(root='data/', train=False, transform=torchvision.transforms.Compose([ToTensor()]))
from collections import deque as de

# I am assuming that
# red is 0
# green is 1
# blue is 2
temp_red = de()
temp_green = de()
temp_blue = de()

for i in dataset:
    temp_red.append(i[0][0])
    temp_green.append(i[0][1])
    temp_blue.append(i[0][2])

temp_red = tuple(temp_red)
temp_green = tuple(temp_green)
temp_blue = tuple(temp_blue)

mean_red = torch.stack(temp_red).mean()
mean_green = torch.stack(temp_green).mean()
mean_blue = torch.stack(temp_blue).mean()

mean_red, mean_green, mean_blue 
# I am assuming that
# red is 0
# green is 1
# blue is 2

std_red = torch.stack(temp_red).std()
std_green = torch.stack(temp_green).std()
std_blue = torch.stack(temp_blue).std()

std_red, std_green, std_blue 
t = torchvision.transforms.Compose([
                                    RandomHorizontalFlip(),
                                    #RandomPerspective(), # too much for a feed forward neural network to handle, will be commented out in the next run
                                    # ColorJitter(brightness=0.27, contrast=0.25, saturation=0.27, hue=0.1),
                                    # RandomGrayscale(), # Caused some images to become just a black screen with no details
                                    ToTensor(),
                                    Normalize((mean_red, mean_green, mean_blue), (std_red, std_green, std_blue)),
                                    # RandomErasing(scale=(0.015, 0.15), ratio=(0.15, 1.5)),
                                   ])

dataset = CIFAR10(root='data/', download=True, transform=t)
test_dataset = CIFAR10(root='data/', train=False, transform=t)
dataset
dataset_size = len(dataset)
dataset_size
test_dataset
test_dataset_size = len(test_dataset)
test_dataset_size
dir(dataset)
dataset.classes
classes = dataset.classes
classes
num_classes = len(dataset.classes)
num_classes
img, label = dataset[0]
img_shape = img.shape
img_shape
import random

random.seed()
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
ind = random.randint(0,50000)

img, label = dataset[ind]
print('Image with permute:', img.permute((1, 2, 0)).shape)
print('Image without permute:', img.shape)
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
count = dict.fromkeys(classes, 0)

for _, label in dataset:
    count[classes[label]] += 1

for i in count:
    print(f"{i} has {count[i]} images")
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name, environment=None)
torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size=256
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)
16 * 16
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,16))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
classes
@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    # This decorator temporarily disables gradient computation for all tensors passed to the function
    @torch.no_grad()  # This is done as we do not want to calculate the gradients of the variables in this function
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    # This decorator temporarily disables gradient computation for all tensors passed to the function
    @torch.no_grad() # This is done as we do not want to calculate the gradients of the variables in this function
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        if epoch%5 == 0:
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
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
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
input_size = 3*32*32
output_size = 10
# LAYER_1 = 64
# LAYER_2 = 32
# LAYER_3 = 32
# LAYER_4 = 16

LAYER_1 = 3072
LAYER_2 = 1536
LAYER_3 = 768
LAYER_4 = 384
LAYER_5 = 128
LAYER_6 = 64

NoOfLayers = 6
# class CIFAR10Model(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.complete_model = nn.Sequential(
#                                   nn.Linear(input_size, LAYER_1),
#                                   nn.ReLU(), # nn.PReLU(),
#                                   nn.Linear(LAYER_1, LAYER_2),
#                                   nn.ReLU(), # nn.PReLU(),
#                                   nn.Linear(LAYER_2, LAYER_3),
#                                   nn.ReLU(), # nn.Softplus(),
#                                   nn.Linear(LAYER_3, LAYER_4),
#                                   nn.ReLU(),
#                                   nn.Linear(LAYER_4, output_size),
#                               )
        
#     def forward(self, xb):
#         # Flatten images into vectors
#         out = xb.view(xb.size(0), -1)
#         # Apply layers & activation functions
#         out = self.complete_model(out)
#         return out
class CIFAR10Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.complete_model = nn.Sequential(
                                  nn.Linear(input_size, LAYER_1),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_1, LAYER_2),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_2, LAYER_3),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_3, LAYER_4),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_4, LAYER_5),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_5, LAYER_6),
                                  nn.ReLU(),
                                  nn.Linear(LAYER_6, output_size)
                              )
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.complete_model(out)
        return out
model = to_device(CIFAR10Model(), device)
history = [evaluate(model, val_loader)]
history
# sessions = [
#     [20, 0.1],
#     [5, 0.01],
#     [5, 0.001],
#     [10, 0.0001]
# ]
# sessions = [
#     [25, 1e-1],
#     [10, 1e-2],
#     [15, 1e-3],
#     [20, 1e-4],
#     [20, 1e-5],
#     [20, 1e-6],
# ]
sessions = [
    [20, 1e-1],
    [60, 5e-4],
#     [30, 5e-4],
#     [20, 2e-4],
#     [20, 25e-6],
#     [20, 5e-6],
]
ctr = 1
for epochs, lr in sessions:
    print("="*80, end='\n\n')
    print(f"Currently in session {ctr}")
    print(f"Current learning rate: {lr}")
    print(f"Number of epochs to run for: {epochs}")
    history += fit(epochs, lr, model, train_loader, val_loader)
    ctr += 1
    print('\n')
# history += fit(???, ???, model, train_loader, val_loader)
# history += fit(???, ???, model, train_loader, val_loader)
# history += fit(???, ???, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
result = evaluate(model, test_loader)
result
if NoOfLayers == 4:
    arch = f"{NoOfLayers} layers ({LAYER_1}, {LAYER_2}, {LAYER_3}, {LAYER_4}, 10)"
elif NoOfLayers == 5:
    arch = f"{NoOfLayers} layers ({LAYER_1}, {LAYER_2}, {LAYER_3}, {LAYER_4}, {LAYER_5}, 10)"
elif NoOfLayers == 6:
    arch = f"{NoOfLayers} layers ({LAYER_1}, {LAYER_2}, {LAYER_3}, {LAYER_4}, {LAYER_5}, {LAYER_6}, 10)"

arch
lrs = [x[0] for x in sessions]
epochs = [x[1] for x in sessions]
test_acc = result['val_acc']
test_loss = result['val_loss']
test_acc, test_loss
torch.save(model.state_dict(), 'cifar10-feedforward.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['cifar10-feedforward.pth'], environment=None)
