# importing the necessary libraries
import os
from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import pathlib
%matplotlib inline
data_dir = "../input/intel-image-classification"
train_dir = data_dir + "/seg_train/seg_train"
test_dir = data_dir + "/seg_test/seg_test"
pred_dir = data_dir + "/seg_pred"
# lets look at the classes in the dataset
classes = os.listdir(train_dir)
classes.sort()
print(classes)
train_transform = T.Compose([T.Resize((150,150)), T.ColorJitter(brightness = 1e-1, contrast = 1e-1, saturation = 1e-1, hue = 1e-1), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.1,0.1,0.1),(1,1,1))])
validation_transform = T.Compose([T.Resize((150,150)), T.ToTensor(), T.Normalize((0.1,0.1,0.1),(1,1,1))])
# normalize has been used randomly

# applying the transforms
train_ds = ImageFolder(train_dir, train_transform)
val_ds = ImageFolder(test_dir, validation_transform)

print(len(train_ds))
print(len(val_ds))
train_ds.class_to_idx
train_ds.targets
batch_size = 150

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 3, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers = 3, pin_memory = True)
def show_batch(dl):
    for images, labels in dl: 
        
        fig, ax = plt.subplots(figsize = (12, 12))
        
        ax.set_xticks([]); ax.set_yticks([])
        
        ax.imshow(make_grid(images[:64], nrow = 8).permute(1, 2, 0))
        
        break
        
show_batch(train_dl)
train_iter = iter(train_dl)
print(type(train_iter))

images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
def accuracy(outputs, labels):
    
    _, preds = torch.max(outputs, dim = 1)
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
        
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

class LandscapeCNN(ImageClassificationBase):
    
    def __init__(self):
        
        super().__init__()
        
        # Using a pretrained model
        self.network = models.resnet50(pretrained = True)
        
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    
    def forward(self, inpt):
        return torch.sigmoid(self.network(inpt))
    
    
    # freeze function trains the fully connected layer to make predictions
    def freeze(self):
        
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
            
        for param in self.network.fc.parameters():
            param.require_grad = True
            
            
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
model = LandscapeCNN()
model
def get_default_device():
    
    #Pick GPU if available, else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    
    #Move tensor(s) to chosen device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    
    #Wrap a dataloader to move data to a device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        #Yield a batch of data after moving it to device
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        #Number of batches
        return len(self.dl)
# getting the gpu 
device = get_default_device()
device
# transferring the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

# transferring the model to the GPU
model = to_device(LandscapeCNN(), device)
# making random predictions
def try_batch(dl):
    
    for images, labels in dl:
        
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(train_dl)
from tqdm.notebook import tqdm
@torch.no_grad()
def evaluate(model, val_loader):
    
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    
    for param_group in optimizer.param_groups:
        return param_group['lr']  
    

def fit(epochs, max_lr, model, train_loader, val_loader, opt_func, decay, grad_clip = None):
    
    # freeing up space on the GPU
    torch.cuda.empty_cache()
    history = []
    
    # defining the optimizer
    optimizer = opt_func(model.parameters(), lr = max_lr, weight_decay = decay)
    
    # defining the scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs = epochs, steps_per_epoch = len(train_loader))
    
    for epoch in range(epochs):
        
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        
        for batch in tqdm(train_loader):
            
            # calculating the loss and computing gradients
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # using gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # updating the lrs for the epochs
            lrs.append(get_lr(optimizer))
            sched.step()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history
torch.cuda.empty_cache()
# evaluating the model which is randomized
evaluate(model, val_dl)
# freezing the model initially
model.freeze()
max_lr = 1e-2
epochs = 4
opt_func = torch.optim.Adamax
decay = 1e-4
grad_clip = 1e-1
history = []
%%time
history += fit(epochs, max_lr, model, train_dl, val_dl, opt_func, decay, grad_clip)
# unfreezing the model
model.unfreeze()
epochs = 5
%%time
history += fit(epochs, max_lr, model, train_dl, val_dl, opt_func, decay, grad_clip)
max_lr = 1e-3
epochs = 3
history += fit(epochs, max_lr, model, train_dl, val_dl, opt_func, decay, grad_clip)
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
def show_sample(img):
    plt.imshow(img.permute(1, 2, 0))
    
    
# function to predict a single image
def predict(image):
    
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    
    max_ = torch.max(preds[0]).item()
    prediction = classes[(preds[0] == max_).nonzero()]
    
    print("Prediction: ", prediction)
    show_sample(image)
pred_ds = ImageFolder(pred_dir, validation_transform)

print(len(pred_ds))
image,target = pred_ds[0]
print(image.shape)
predict(pred_ds[12][0])
predict(pred_ds[3][0])
predict(pred_ds[9][0])
predict(pred_ds[2][0])
predict(pred_ds[13][0])
# getting the dataloader from the dataset
test_dl = DeviceDataLoader(DataLoader(pred_ds, batch_size, num_workers = 3, pin_memory = True), device)
# getting the predictions for the entire test data loader
def predict_dl(dl, model):
    
    torch.cuda.empty_cache()
    batch_probs = []
    
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
        
    batch_probs = torch.cat(batch_probs)
    return [x for x in batch_probs]
test_preds = predict_dl(test_dl, model)
# saving the predictions to the output folder
submissions = pd.DataFrame()
submissions.Label = test_preds
submissions.head()
# storing our submission into a .csv file
submissions.to_csv("predictions.csv")
!pip install jovian --upgrade

import jovian

jovian.commit(project = "course-project")