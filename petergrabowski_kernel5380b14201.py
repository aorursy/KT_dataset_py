project_name = 'project-piotr-grabowski'
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name, enviroment=None)
import os
import gc
import torch
import torchvision
import tarfile
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
classes = os.listdir('../input/tomato/project-data/train')
print(classes)
tomatoes = ImageFolder('../input/tomato/project-data')
tomato_img_size = []
for images, labels in tomatoes:
    tomato_img_size.append(images.size)

x = []
y = []
for img_sizes in tomato_img_size:
    x.append(img_sizes[0])
    y.append(img_sizes[1])

markerline, stemlines, baseline = plt.stem(x, y, linefmt='grey', markerfmt='D', bottom=140, use_line_collection=True)
plt.xlabel('Pixel size(x)')
plt.ylabel('Pixel size(y)')
plt.title('Tomato image sizes in pixels')
markerline.set_markerfacecolor('none')
plt.show()
train_tfms = tt.Compose([tt.Resize((64,64)),tt.ToTensor()])
val_tfms = tt.Compose([tt.Resize((64,64)),tt.ToTensor()])

train_ds = ImageFolder('../input/tomato/project-data/train', train_tfms)
val_ds = ImageFolder('../input/tomato/project-data/test', val_tfms)
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
show_batch(train_dl)
input_size = 64 * 64 * 3
num_classes = 2
class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 12288)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss, 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss' : epoch_loss.item(), 'val_acc' : epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss : {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = LogitModel()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        #Training phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
result = evaluate(model, val_dl)
result
history1 = fit(5, 0.001, model, train_dl, val_dl)
history2 = fit(5, 0.0001, model, train_dl, val_dl)
history3 = fit(5, 0.00001, model, train_dl, val_dl)
history4 = fit(5, 0.00001, model, train_dl, val_dl)
history = [result] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
end_result = evaluate(model, val_dl)
print(end_result)
test_tfms = tt.Compose([tt.Resize((64,64)),tt.ToTensor()])
test_dataset = ImageFolder('../input/tomato/project-data/test', test_tfms)
len(test_dataset)
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return test_dataset.classes[preds[0].item()]
img, label = test_dataset[0]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicted: ', predict_image(img, model))
img, label = test_dataset[12]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicted: ', predict_image(img, model))
img, label = test_dataset[90]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicted: ', predict_image(img, model))
# Clear previously recorded hyperparams & metrics
jovian.reset()
torch.save(model.state_dict(), 'project-piotr-grabowski.pth')
lrs = [0.001,0.0001,0.00001,00000.1]
epochs = [5,5,5,5,5]
val_loss, val_acc = result
jovian.log_hyperparams(lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(val_loss=val_loss, val_acc=val_acc)
jovian.commit(project=project_name, outputs=['project-piotr-grabowski.pth'], environment=None)
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class FfNnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        hidden_size = hidden_size * 2
        self.linear3 = nn.Linear(hidden_size, hidden_size * 2)
        hidden_size = hidden_size * 2
        self.linear4 = nn.Linear(hidden_size, hidden_size * 2)
        hidden_size = hidden_size * 2
        self.linear5 = nn.Linear(hidden_size, output_size)
        
    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        #Layer 1
        out = self.linear1(out)
        out = F.relu(out)
        #Layer 2
        out = self.linear2(out)
        out = F.relu(out)
        #Layer 3
        out = self.linear3(out)
        out = F.relu(out)
        #Layer 4
        out = self.linear4(out)
        out = F.relu(out)
        #Layer 5
        out = self.linear5(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss, 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss' : epoch_loss.item(), 'val_acc' : epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
input_size = 64 * 64 * 3
hidden_size = 432
num_classes = 2
@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        # Training phase
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
model = FfNnModel(input_size, hidden_size, num_classes)
model = to_device(model, device)
history = [evaluate(model, val_dl)]
history
history += fit(25, 0.03, model, train_dl, val_dl)
losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
plot_losses(history)
test_tfms = tt.Compose([tt.Resize((64,64)),tt.ToTensor()])
test_dataset = ImageFolder('../input/tomato/project-data/test', test_tfms)
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return train_ds.classes[preds[0].item()]
img, label = test_dataset[0]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicted: ', predict_image(img, model))
img, label = test_dataset[2]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicited: ', predict_image(img, model))
img, label = test_dataset[14]
plt.imshow(img.permute(1,2,0))
print('Label:', train_ds.classes[label], ', Predicted: ', predict_image(img, model))
uploaded_model = FfNnModel(input_size, hidden_size, num_classes)
uploaded_model.load_state_dict(torch.load('../input/feedforwardnetworkprojecttomatoes/project-piotr-grabowski.pth', map_location = torch.device('cuda')))
uploaded_model.state_dict()
arch = print(model)
lrs = [0.03]
epochs = [25]
print(history[len(history) - 1])
test_loss = 2.7377545833587646
test_acc = 0.7204300165176392
torch.save(uploaded_model.state_dict(), 'project-piotr-grabowski.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['project-piotr-grabowski.pth'], environment=None)
batch_size = 3
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss.detach(), 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss' : epoch_loss.item(), 'val_acc' : epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class ProjectCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output: 128 * 32 * 32
        
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output: 512 * 16 * 16
        
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output: 2048 * 8 * 8
        
        
            nn.Flatten(),
            nn.Linear(2048 * 8 * 8, 8192),
            nn.ReLU(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 2))
        
    def forward(self, xb):
        return self.network(xb)
model = ProjectCnnModel()
model
model = to_device(ProjectCnnModel(), device)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        #Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
evaluate(model, val_dl)
history = fit(60, 0.03, model, train_dl, val_dl, torch.optim.SGD)
del train_dl, val_dl
del model
gc.collect()
gc.collect()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.empty_cache()
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
plot_accuracies(history)
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training','Validation'])
    plt.title('Loss vs. No. of epochs')
plot_losses(history)
test_tfms = tt.Compose([tt.Resize((64,64)),tt.ToTensor()])
test_dataset = ImageFolder('../input/tomato/project-data/test', test_tfms)
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return test_dataset.classes[preds[0].item()]
img, label = test_dataset[0]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[10]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[89]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
uploaded_model = FfNnModel(input_size, hidden_size, num_classes)
uploaded_model.load_state_dict(torch.load('../input/feedforwardnetworkprojecttomatoes/project-piotr-grabowski.pth', map_location = torch.device('cuda')))
uploaded_model.state_dict()
model_arch = print(model)
arch = [model_arch]
lrs = [0.03]
epochs = [60]
print(history[len(history) - 1])
test_loss = 2.7377545833587646
test_acc = 0.7204300165176392
torch.save(uploaded_model.state_dict(), 'project-piotr-grabowski.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['project-piotr-grabowski.pth'], environment=None)
stats = ((0.4914,0.4822,0.7465), (0.1011, 0.1992, 0.2020))
train_tfms = tt.Compose([tt.Resize((128,128)),
                        tt.RandomCrop(128, padding=4, padding_mode='symmetric'),
                        tt.RandomRotation(15),
                        tt.RandomHorizontalFlip(),
                        tt.ToTensor(),
                        tt.Normalize(*stats, inplace=True)])
                        #])
val_tfms = tt.Compose([tt.Resize((128,128)),tt.ToTensor(), tt.Normalize(*stats)])
train_ds = ImageFolder('../input/tomato/project-data/train', train_tfms)
val_ds = ImageFolder('../input/tomato/project-data/test', val_tfms)
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:128], nrow=8).permute(1, 2, 0))
        break
show_batch(train_dl)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def convolution_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = convolution_block(in_channels, 54)
        self.conv2 = convolution_block(54, 108, pool=True)
        self.res1 = nn.Sequential(convolution_block(108,108), convolution_block(108,108))
        
        self.conv3 = convolution_block(108, 216, pool=True)
        self.conv4 = convolution_block(216, 432, pool=True)
        self.res2 = nn.Sequential(convolution_block(432,432), convolution_block(432,432))
        
        self.conv5 = convolution_block(432, 864, pool=True)
        self.conv6 = convolution_block(864, 1728, pool=True)
        self.res3 = nn.Sequential(convolution_block(1728,1728), convolution_block(1728,1728))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(1728, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss' : loss.detach(), 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss' : epoch_loss.item(), 'val_acc' : epoch_acc.item()}
        
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.7f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
model = to_device(ResNet9(3, 2), device)
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
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        #Training phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            #Gradient Clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            #Record and update learining rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
        
        #Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
history = [evaluate(model, val_dl)]
history
epochs = 54
max_lr = 0.0002
grad_clip = 0.1
weight_decay = 1e-3
opt_func = torch.optim.SGD
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, grad_clip, weight_decay, opt_func=opt_func)
def plot_accuracies(history):
    accuracies = [x.get('val_acc') for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
plot_accuracies(history)
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training','Validation'])
    plt.title('Loss vs. No. of epochs')
plot_losses(history)
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs',[]) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learing Rate vs. Batch no.')
plot_lrs(history)
test_tfms = tt.Compose([tt.Resize((128,128)),tt.ToTensor()])
test_dataset = ImageFolder('../input/tomato/project-data/test', test_tfms)
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return test_dataset.classes[preds[0].item()]    
img, label = test_dataset[88]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[25]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[80]
plt.imshow(img.permute(1,2,0))
print('Label:', test_dataset.classes[label], ', Predicted:', predict_image(img, model))
uploaded_model = FfNnModel(input_size, hidden_size, num_classes)
uploaded_model.load_state_dict(torch.load('../input/feedforwardnetworkprojecttomatoes/project-piotr-grabowski.pth', map_location = torch.device('cuda')))
uploaded_model.state_dict()
model_arch = print(model)
arch = [model_arch]
lrs = print(np.concatenate([x.get('lrs',[]) for x in history]))
epochs = 54
print(history[len(history) - 1])
test_acc = 0.8709676861763
test_loss = 0.4150371253490448
torch.save(uploaded_model.state_dict(), 'project-piotr-grabowski.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['project-piotr-grabowski.pth'], environment=None)