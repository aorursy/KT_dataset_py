# imports
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from PIL import Image
# Unzip the data
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip", "r") as z:
    z.extractall(".")
train_dir = 'train'
test_dir = 'test1'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
# Implement the Dataset class
class CatDogDataset(Dataset):
    def __init__(self, file_list, directory, mode='train', transform=None):
        self.file_list = file_list
        self.dir = directory
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            return img.numpy().astype('float32'), self.label
        else:
            return img.numpy().astype('float32'), self.file_list[idx]
        
# Prepare train dataset
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.Resize(128),
    transforms.ToTensor()
])

cat_files = [file for file in train_files if 'cat' in file]
dog_files = [file for file in train_files if 'dog' in file]

cats = CatDogDataset(cat_files, train_dir, mode='train', transform=data_transform)
dogs = CatDogDataset(dog_files, train_dir, mode='train', transform=data_transform)

train_data = ConcatDataset([cats, dogs])
# Prepare the dataloader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
dataiter = iter(train_loader)
images, labels = dataiter.next()
# Plot the images
fig = plt.figure(figsize=(16, 24))
grid_imgs = torchvision.utils.make_grid(images[:24])
np_grid_imgs = grid_imgs.numpy()
# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
images.shape, labels.shape # (batch_size, channels, hight, width), (batch_size, 1)
# Download a pre-trained model
device = 'cuda'
model = torchvision.models.resnet152(pretrained=True)
# Let's see how does the architecture look
model 
# Freeze the model parameters for transfer learning
# for param in model.parameters():
#     param.requires_grad = False
import torch.nn as nn
from torch.optim import Adam
# Define and Update the last fully-connected layer
input_size = model.fc.in_features
output_size = 2
model.fc = nn.Sequential(
    nn.Linear(input_size, 1000),
    nn.Linear(1000, 500),
    nn.Linear(500, output_size))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
# Model training
num_epochs = 2
train_loss = 0
print_every = 20
counter = 1
loss_list = []
acc_list = []
model.train
for i in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % print_every == 0:
            predicted_labels = torch.argmax(preds, dim=1)
            corrects = predicted_labels.eq(labels)
            accuracy = torch.mean(corrects.float())
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'\
                  .format(i+1, num_epochs, counter, train_loss/print_every, accuracy))
            loss_list.append(train_loss/print_every)
            acc_list.append(accuracy)
            train_loss = 0
        counter += 1
        
# Plot the training history
plt.plot(loss_list, label='loss')
plt.plot(acc_list, label='accuracy')
plt.legend()
plt.title('training loss and accuracy')
plt.show()
model_path = 'ckpt_resnet152_catdog.pth'
torch.save(model.state_dict(), model_path)
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

testset = CatDogDataset(test_files, test_dir, mode='test', transform = test_transform)
testloader = DataLoader(testset, batch_size = 64, shuffle=False, num_workers=4)
samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
ad = {0:'cat', 1:'dog'}
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
