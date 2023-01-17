import os 
import torch  
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

%matplotlib inline
train_dataset_dir = '../input/intel-image-classification/seg_train/seg_train'
test_dataset_dir = '../input/intel-image-classification/seg_test/seg_test'

dataset_dir = ['../input/intel-image-classification/seg_train/seg_train',
               '../input/intel-image-classification/seg_test/seg_test']
label_list=['buildings','forest','glacier','mountain','sea','street']

labels = {label_list:i for i,label_list in enumerate(label_list)}
labels

def create_dataset(dataset_path,labels,file_name):
    labelled_arr=np.array([])
    for subdir, label in labels.items():
        img_dir = os.path.join(dataset_path, subdir) 
        files = np.array(os.listdir(img_dir)).reshape(-1,1) 
        target = np.array([label for i in range(files.shape[0])]).reshape(-1,1) 
        data = np.concatenate((files, target), axis = 1) 
        labelled_arr = np.append(labelled_arr, data)
    labelled_arr = labelled_arr.reshape(-1,2)

    dataframe = pd.DataFrame(labelled_arr)
    dataframe.columns = ['image', 'label']
    dataframe['label'] = dataframe['label'].astype('int')
    print (dataframe.head())
    print (file_name)
    dataframe.to_csv(file_name, index = False)
    return dataframe
    
train_df = create_dataset(train_dataset_dir,labels,'./train.csv')
test_df = create_dataset(test_dataset_dir, labels,'./test.csv')
class IntelImageDataset(Dataset):
    def __init__(self, dataframe, data_dir, label_dict, transform = None):
        self.df = dataframe
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, label = self.df.loc[idx]
        class_labels = list(self.label_dict.keys())
        img_path = self.data_dir + '/' + class_labels[label] + '/' + img_name
        img = img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label 
train_tfms = T.Compose([
    T.Resize([256,256]),
    T.ToTensor()
#     T.RandomErasing()
])

test_tfms = T.Compose([
    T.Resize([256, 256]),
    T.ToTensor(),
])
train_ds = IntelImageDataset(train_df, train_dataset_dir, labels, transform = train_tfms)


test_ds = IntelImageDataset(test_df, test_dataset_dir, labels, transform = test_tfms)
len(train_ds), len(test_ds)
def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', target)
show_sample(*train_ds[14000],invert=False)
val_size = 4000
train_size = len(train_ds) - val_size

train_ds, val_ds = random_split(train_ds, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size=32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, 
                    num_workers=2, pin_memory=True)
simple_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2, 2)
)
for images, labels in train_dl:
    print('images.shape:', images.shape)
    print (labels)
    out = simple_model(images)
    print('out.shape:', out.shape)
    break
def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=8).permute(1, 2, 0))
        break
show_batch(train_dl, invert=True)
show_batch(train_dl, invert=False)
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

def accuracy(outputs, target):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == target).item() / len(preds))
class IntelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        score = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))
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
class FNNModel(IntelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(3*256*256, 512)
        # hidden layer2
        self.linear2 = nn.Linear(512,256)
        # hidden layer3
        self.linear3 = nn.Linear(256,128)
        # hidden layer4
        self.linear4 = nn.Linear(128,64)
        # output layer
        self.linear5 = nn.Linear(64, 6)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        
        # Get intermediate outputs using hidden layer1
        out = self.linear1(out)
        # Apply activation function
        out = F.relu(out)
        
        # Get intermediate outputs using hidden layer2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        
        # Get intermediate outputs using hidden layer3
        out = self.linear3(out)
        # Apply activation function
        out = F.relu(out)
        
        # Get intermediate outputs using hidden layer4
        out = self.linear4(out)
        # Apply activation function
        out = F.relu(out)
        
        
        # Get predictions using output layer
        out = self.linear5(out)
        return out
model = FNNModel()
model
model = to_device(model,device)
for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print (labels)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break
history = [evaluate(model, val_dl)]
history
history += fit(10, 0.001, model, train_dl, val_dl)
def plot_accuracies(history):
#     print(history)
    accuracies = [x['val_score'] for x in history]
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
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return label_list[preds[0].item()]
img,label = test_ds[0]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model))
img,label = test_ds[10]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model))
class CnnModel(IntelImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 32 x 32

            nn.Flatten(), 
            nn.Linear(256*32*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6))
        
    def forward(self, xb):
        return self.network(xb)
model1 = CnnModel()
model1
model1 = to_device(CnnModel(), device)
for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model1(images)
    print (labels)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break
evaluate(model1, val_dl)
num_epochs = 10
opt_func = torch.optim.Adam
lr = 1e-4
%%time
history1 = fit(num_epochs, lr, model1, train_dl, val_dl, opt_func)
plot_accuracies(history1)
plot_losses(history1)
img,label = test_ds[0]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model1))
img,label = test_ds[10]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model1))

train_tfms = T.Compose([
    T.Resize([256,256]),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(), 
    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize([256,256]), 
    T.ToTensor(),
])
train_ds2 = IntelImageDataset(train_df, train_dataset_dir, labels, transform = train_tfms)


test_ds2 = IntelImageDataset(test_df, test_dataset_dir, labels, transform = test_tfms)
show_sample(*train_ds2[14000],invert=False)
show_sample(*train_ds2[14000],invert=False)
val_size = 4000
train_size = len(train_ds2) - val_size

train_ds2, val_ds2 = random_split(train_ds2, [train_size, val_size])
len(train_ds2), len(val_ds2)

batch_size = 32
train_dl2 = DataLoader(train_ds2, batch_size, shuffle=True, 
                      num_workers=3, pin_memory=True)
val_dl2 = DataLoader(val_ds2, batch_size*2, 
                    num_workers=2, pin_memory=True)
show_batch(train_dl2, invert=True)
class IntelImageResnet(IntelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
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
train_dl2 = DeviceDataLoader(train_dl2, device)
val_dl2 = DeviceDataLoader(val_dl2, device)
to_device(model, device);
@torch.no_grad()
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
model2 = to_device(IntelImageResnet(), device)
history2 = [evaluate(model2, val_dl2)]
history2
model2.freeze()
epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.SGD
%%time
history2 += fit_one_cycle(epochs, max_lr, model2, train_dl2, val_dl2, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
model2.unfreeze()
%%time
history2 += fit_one_cycle(epochs, 0.01, model2, train_dl2, val_dl2, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
plot_accuracies(history2)
plot_losses(history2)
img,label = test_ds[0]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model2))
img,label = test_ds[10]
print (label)
plt.imshow(img.permute(1, 2, 0))
print('Label:', label_list[label], ', Predicted:', predict_image(img, model2))
!pip install jovian --upgrade -q
import jovian
project_name='intelimageclassification'
jovian.commit(project=project_name)
