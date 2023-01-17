#Import necessary packages

import pandas as pd #data manipulation packages
import numpy as np
from sklearn.preprocessing import LabelEncoder #creating a label dictionary 
from sklearn.model_selection import train_test_split #splitting data to train & validation

from torchvision.datasets.folder import default_loader #loading image data
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt #data visualization
import seaborn as sns
from torchvision.utils import make_grid
    
import torchvision.transforms as transforms #machine learning packages
import torch
from torch.utils.data import Dataset, DataLoader

#Check summary file & metadata content

summary_csv = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_dataset_Summary.csv', index_col=0)
total = summary_csv['Image_Count'].sum()
print('Total observations: ', total)
print(summary_csv)
#Loan metadata
labels_csv = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
labels_csv = labels_csv.rename(columns = {'Unnamed: 0':'Index'}) #rename index column

labels_csv.head(10000) #Note that some rowid's do not match with Index
#Create a new category Label Subcat where we recode the categories as Normal, Pneumonia Viral, Pneumonia Bacterial, or Pneumonia Stress.
#We do not model for COVID 19 since we only have a few datasets for COVID.

labels_csv.loc[labels_csv['Label'] == 'Normal', 'Label_Subcat'] = 'Normal'
labels_csv.loc[(labels_csv['Label'] == 'Pnemonia') & (labels_csv['Label_1_Virus_category']=='Virus'), 'Label_Subcat'] = 'Pneumonia, Viral'
labels_csv.loc[(labels_csv['Label'] == 'Pnemonia') & (labels_csv['Label_1_Virus_category']=='bacteria'), 'Label_Subcat'] = 'Pneumonia, Bacterial'
labels_csv.loc[(labels_csv['Label'] == 'Pnemonia') & (labels_csv['Label_1_Virus_category']=='Stress-Smoking'), 'Label_Subcat'] = 'Pneumonia, Stress'  
#Determing share of dataset per category
c = labels_csv.groupby(['Dataset_type', 'Label_Subcat'])['X_ray_image_name'].count().rename("count")
c / c.groupby(level=0).sum() *100
#Create the directory for train and test sets

dir_train = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
dir_test = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'

#Filter dataset to train and test
train_filtered = labels_csv[labels_csv['Dataset_type']=='TRAIN'][['X_ray_image_name','Label_Subcat']]
test_filtered = labels_csv[labels_csv['Dataset_type']=='TEST'][['X_ray_image_name','Label_Subcat']]
#Split train dataset to train and validation data

torch.manual_seed(10) #fix randomization
 
train_data, val_data=train_test_split(train_filtered, stratify=train_filtered["Label_Subcat"], test_size=0.1, random_state=30) #split the data proportionally by category size
test_data= test_filtered
#Determine share of categories for train dataset
t_cnt=train_data.groupby(['Label_Subcat'])['X_ray_image_name'].count().rename("count")
t_cnt / t_cnt.sum() *100
#Determine share of categories for validation dataset
v_cnt=val_data.groupby(['Label_Subcat'])['X_ray_image_name'].count().rename("count")
v_cnt / v_cnt.sum() *100
#Create dictionary for subcategories

labelencoder = LabelEncoder()
label_subcat_list = [subcat for subcat in labels_csv["Label_Subcat"].unique()]
label_subcat_transform = sorted(labelencoder.fit_transform(label_subcat_list))
subcat_dict = {label_subcat_list[l]:label_subcat_transform[l] for l in range(len(label_subcat_list))}
subcat_dict
subcat_dict_items = { x:y for y,x in subcat_dict.items()}
subcat_dict_items
#Create a custom data loader
class ChestXray(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = csv_file
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        img_fname = self.root_dir +  self.df.iloc[idx,0]
        img = default_loader(img_fname)
        actual = self.df.iloc[idx,1]
        actual_dict = subcat_dict[actual]
        if self.transform:
            img = self.transform(img)
        return img, actual_dict
#Load images using sample data to check if data is loaded properly

torch.manual_seed(10)
temp_dataset = ChestXray(csv_file=train_filtered.sample(frac=1), root_dir=dir_train)


fig = plt.figure(figsize=(20, 10))

for i, batch in enumerate(temp_dataset):
    img, img_label = batch
    fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    plt.imshow(img)
    plt.title(subcat_dict_items[img_label])
    if i == 9:
        break
#Resize and convert images to Tensor
transform_image = transforms.Compose([ transforms.Resize((100, 100)), 
                                       transforms.ToTensor()])

#Load datasets using custom loader
train_data_cload = ChestXray(train_data, root_dir=dir_train, transform=transform_image)
val_data_cload = ChestXray(val_data, root_dir=dir_train, transform=transform_image)
test_data_cload  = ChestXray(test_data, root_dir=dir_test, transform=transform_image)

# create dataloaders
from torch.utils.data import DataLoader

batch_size=32
num_workers=4

train_dataset = DataLoader(train_data_cload, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
val_dataset = DataLoader(val_data_cload, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
test_dataset = DataLoader(test_data_cload, batch_size=batch_size*2, shuffle=True, pin_memory=True, num_workers=num_workers)

#Plot sample images from train dataset
for images, _ in train_dataset:
    print('images.shape:', images.shape)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
! pip install jovian
import jovian
project_name="Course-Project"
# jovian.commit(message="getting-data-ready", project=project_name)
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
for images, labels in train_dataset:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break
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
train_dataset = DeviceDataLoader(train_dataset, device)
val_dataset = DeviceDataLoader(val_dataset, device)
test_dataset = DeviceDataLoader(test_dataset, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# class XRayModel(nn.Module):
#     """Feedfoward neural network with 1 hidden layer"""
#     def __init__(self, in_size, hidden_size, out_size):
#         super().__init__()
#         # hidden layer
#         self.linear1 = nn.Linear(in_size, hidden_size)
#         # output layer
#         self.linear2 = nn.Linear(hidden_size, out_size)
        
#     def forward(self, xb):
#         # Flatten the image tensors
#         xb = xb.view(xb.size(0), -1)
#         # Get intermediate outputs using hidden layer
#         out = self.linear1(xb)
#         # Apply activation function
#         out = F.relu(out)
#         # Get predictions using output layer
#         out = self.linear2(out)
#         return out
    
#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
    
#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss, 'val_acc': acc}
        
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
# input_size = 30000
# hidden_size = 32 # you can change this
# num_classes = 4
# model = XRayModel(input_size, hidden_size=hidden_size, out_size=num_classes)
# to_device(model, device)
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
class XRayModelFF3(ImageClassificationBase):
    def __init__(self):
        super().__init__()
         # hidden layer
        self.linear1 = nn.Linear(input_size, 7048)
        # hidden layer 2
        self.linear2 = nn.Linear(7048, 784)
        # hidden layer 3
        self.linear3 = nn.Linear(784, 587)
         # output layer
        self.linear4 = nn.Linear(587, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
       # Get intermediate outputs using hidden layer 1
        out = self.linear1(out)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 3
        out = self.linear3(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear4(out)
        
        return out
input_size = 30000
output_size = 4
model = XRayModelFF3()
to_device(model, device)
# class XRayCNN(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

#             nn.Flatten(), 
#             nn.Linear(36864, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 4))
        
#     def forward(self, xb):
#         return self.network(xb)
# model = XRayCNN()
# to_device(model, device)
for t in model.parameters():
    print(t.shape)
for images, labels in train_dataset:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
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
history = []
history = [evaluate(model, val_dataset)]
history
history1 = fit(5, 0.0005, model, train_dataset, val_dataset)
history2 = fit(10, 0.001, model, train_dataset, val_dataset)
history3 = fit(5, 0.000001, model, train_dataset, val_dataset)
# history += fit(10, 0.05, model, train_dataset, val_dataset)
# history += fit(10, 0.99, model, train_dataset, val_dataset)
history = history1 + history2 + history3
plot_losses(history)
plot_accuracies(history)
evaluate(model, test_dataset)
test_acc = 39.11
test_loss = 1.17
arch = "Feedforward 3 hidden layers (7048, 784, 587)"
lrs = [0.0005,0.001,0.000001]
epochs = [5,10,5]
torch.save(model.state_dict(), 'xray-feedforward-3layers')
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['xray-feedforward.pth'], environment=None)
