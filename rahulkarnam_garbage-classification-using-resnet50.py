import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import glob
import torchvision.transforms as transforms
import torch.optim as optim
data_dir = '../input/garbage-classification/garbage classification/Garbage classification'

print(os.listdir(data_dir))
classes = os.listdir(data_dir)
print("\nClasses:", classes)
print("\nNumber of Classes:", len(classes))
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
transformations = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                       ])
dataset = ImageFolder(data_dir, transform=transformations)
print(len(dataset))
img, label = dataset[0]
print(img.shape, label)
img
import matplotlib.pyplot as plt
%matplotlib inline

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
img, label = dataset[0]
show_example(img, label)
show_example(*dataset[2000])
random_seed = 42
torch.manual_seed(random_seed);
len(dataset)
val_per = 0.1
train_size = len(dataset) - int(val_per*len(dataset))
val_size = int(val_per*len(dataset))

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
from torch.utils.data.dataloader import DataLoader

batch_size= 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)
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
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 87)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
model = ResNet()
model
for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break
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
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
model = to_device(ResNet(), device)
evaluate(model, val_dl)
num_epochs = 3
opt_func = torch.optim.Adam
lr = 5.5e-5
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
num_epochs = 25
opt_func = torch.optim.Adam
lr = 5.5e-5
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
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
base_path='../input/test-garbage'
test_dataset = ImageFolder(base_path+'/TEST', transform=transformations)
test_dataset
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]
img, label = test_dataset[1]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[2]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[2]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
result
torch.save(model.state_dict(), 'waste-classification-data.pth')
import urllib.request
# urllib.request.urlretrieve("data:image/jpeg;base64,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", "cardboard.jpg")
urllib.request.urlretrieve("data:image/jpeg;base64,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", "cardboard.jpg")
loaded_model = model
loaded_model.load_state_dict(torch.load('./waste-classification-data.pth'))
from PIL import Image
from pathlib import Path
image = Image.open(Path('./cardboard.jpg'))

example_image = transformations(image)
plt.imshow(example_image.permute(1, 2, 0))
predict_image(example_image, loaded_model)
