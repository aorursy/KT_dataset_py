# # Uncomment and run the commands below if imports fail
# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
# Project name used for jovian.commit
project_name = '03-cifar10-feedforward'
dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
dataset_size = len(dataset)
dataset_size
test_dataset_size = len(test_dataset)
test_dataset_size
classes = dataset.classes
classes
num_classes = len(classes)
num_classes
img, label = dataset[0]
img_shape = img.shape
img_shape
img, label = dataset[0]
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
import pandas as pd
pd.Series(data=dataset.targets).value_counts()
dataset.class_to_idx
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name, environment=None)
torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size=128
# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
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
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
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
    plt.ylim(0, 1)
    plt.title('Accuracy vs. No. of epochs');
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
input_size = 3*32*32
output_size = 10
# hidden_size = 16
# class CIFAR10Model(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)
    
#     def forward(self, xb):
#         out = xb.view(xb.size(0), -1)
#         out = self.linear1(out)
#         out = F.relu(out) 
#         out = self.linear2(out)
#         return out


# class CIFAR10Model(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 1536)
#         self.fc2 = nn.Linear(1536, 768)
#         self.fc3 = nn.Linear(768, 384)
#         self.fc4 = nn.Linear(384, 128)
#         self.fc5 = nn.Linear(128, output_size)
        
#     def forward(self, xb):
#         # Flatten images to vectors
#         out = xb.view(xb.size(0), -1)
#         # Add layers
#         out = self.fc1(out)
#         out = F.relu(out)
#         out = self.fc2(out)
#         out = F.relu(out)
#         out = self.fc3(out)
#         out = F.relu(out)
#         out = self.fc4(out)
#         out = F.relu(out)
#         out = self.fc5(out)
#         return out
class CIFAR10Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 2000)
        self.linear2 = nn.Linear(2000, 1000)
        self.linear3 = nn.Linear(1000, 500)
        self.linear4 = nn.Linear(500, 200)
        self.linear5 = nn.Linear(200, output_size)
    
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
model = to_device(CIFAR10Model(), device)
model
%%time

history = [evaluate(model, val_loader)]
history
%%time

lrs = [0.5, 0.4]
epochs = [5, 5]

for lr, nb_epochs in zip(lrs, epochs):
    print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
    history += fit(nb_epochs, lr, model, train_loader, val_loader)
%%time

lrs = [0.3, 0.2]
epochs = [5, 5]

for lr, nb_epochs in zip(lrs, epochs):
    print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
    history += fit(nb_epochs, lr, model, train_loader, val_loader)
%%time

lrs = [0.1, 0.05]
epochs = [5, 5]

for lr, nb_epochs in zip(lrs, epochs):
    print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
    history += fit(nb_epochs, lr, model, train_loader, val_loader)
import numpy as np

np.arange(0.5, 0.1, -0.1)
%%time

lrs = [0.01, 0.001]
epochs = [5, 5]

for lr, nb_epochs in zip(lrs, epochs):
    print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
    history += fit(nb_epochs, lr, model, train_loader, val_loader)
%%time

lrs = [0.0005, 0.0001]
epochs = [5, 5]

for lr, nb_epochs in zip(lrs, epochs):
    print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
    history += fit(nb_epochs, lr, model, train_loader, val_loader)
# %%time

# print("Let's suddenly increase learning rates and see the loss and accuracy")

# lrs = [0.35, 0.25]
# epochs = [5, 5]

# for lr, nb_epochs in zip(lrs, epochs):
#     print(f"\nLearningRate: {lr} \t TotalEpochs: {nb_epochs}")
#     history += fit(nb_epochs, lr, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
model_evaluation = evaluate(model, test_loader)
model_evaluation
arch = "5 Layers (2000, 1000, 500, 200, 10)"
test_acc = model_evaluation['val_acc']
test_loss = model_evaluation['val_loss']
print(f"test_acc: {test_acc}\ntest_loss: {test_loss}")
torch.save(model.state_dict(), 'cifar10-feedforward.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
lrs_history = [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0005, 0.0001]
epochs_history = [5] * len(lrs_history)

jovian.log_hyperparams(arch=arch, 
                       lrs=lrs_history, 
                       epochs=epochs_history)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(message="Nishant - Assignment #3", project=project_name, outputs=['cifar10-feedforward.pth'], environment=None)
