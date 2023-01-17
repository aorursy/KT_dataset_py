import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
## loading the train dataset

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_df.head()
## loading the test dataset

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df.head()
print(f"Train dataframe size: {len(train_df)}")
print(f"Test dataframe size: {len(test_df)}")
## Visualizing an image from the train set

test_image = np.array(train_df.loc[:,'pixel0':])[20].reshape(28,28) ### getting the 20th image from the train dataset

plt.imshow(test_image, cmap="gray")
plt.show()
print("Label as in the train data: ",train_df['label'][20])
X = train_df.loc[:,'pixel0':]
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

print(f"X_train size: {X_train.shape} and y_train size = {y_train.shape}")
print(f"X_val size: {X_val.shape} and y_val size = {y_val.shape}")
print(f"X_train type: {type(X_train)}")
## loading the test dataframe

X_test = test_df.loc[:,:]

print(f"X_test size: {X_test.shape}")
class MNistDataset(Dataset):
    '''
        Custom pytorch dataset to read and transform MNist data from csv.
    '''
    def __init__(self, data, train=True, transform=None):
        self.data = data
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return self.data[0].values.shape[0]
    
    def __getitem__(self, index):
        X = self.data[0].values[index].astype(np.uint8).reshape((28,28))
        
        if self.train:
            y = self.data[1].values[index].astype(np.long)
        
        if self.transform:
            X = self.transform(X)
        
        if self.train:
            return X,y
        else:
            return X
        
transform = transforms.ToTensor() ## This transformation converts np arrays to tensors
train_ds = MNistDataset((X_train,y_train), train=True, transform=transform)
val_ds = MNistDataset((X_val,y_val), train=True, transform=transform) ## train=True; since this is a split of train dataframe.
test_ds = MNistDataset((X_test,), train=False, transform=transform)
print(f"Size of the Train dataset: {len(train_ds)}")
print(f"Size of the Validation dataset: {len(val_ds)}")
print(f"Size of the Test dataset: {len(test_ds)}")
## random check of the dataset created

image, label = train_ds[120]
val_image, val_label = val_ds[120]
test_image = test_ds[120]

print(f'Image shape: {image.shape} and Type : {type(image)}  ')
print(f'Label: {label} \t Label Type : {type(label)}')
print(f'Validation Image shape: {val_image.shape} and Validation Image Type : {type(val_image)}')
print(f'Label: {val_label} \t Label Type : {type(val_label)}')

print(f'Test Image shape: {test_image.shape} and Test Image Type : {type(test_image)}')

## sample row display
train_ds[120]
train_loader = DataLoader(train_ds, shuffle=True, batch_size=100)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=100)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=28000)  ## we don't need to shuffle test data's. Using the complete volume as batch size as iteration is not required.
class MNistClassifierModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(1,6,3,1) ## params: in_channels, out_channels, kernel_size, stride
        self.conv2d2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
    
    
    def forward(self,x):
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,5*5*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
model = MNistClassifierModel()
model
criterions = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  ## using Adam optimizer with Learning rate: 0.001
epochs = 8

train_losses = []
val_losses = []

train_correct = []
val_correct = []

for i in range(epochs):
    
    train_corr = 0
    val_corr = 0
    
    ## iterating over the training dataset
    for b, (X_train, y_train) in enumerate(train_loader):
        
        b+=1
        
        # predicting from the model and calculating loss
        y_pred = model(X_train)
        losses = criterions(y_pred, y_train)
        
        predicted_label = torch.max(y_pred.data,1)[1]
        train_corr += (predicted_label == y_train).sum()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # printing after every 100 batches of execution
        if b%100 == 0:
            print(f"Epoch:{i} \t Batch:{b} [{b*100}/33600] \t Loss:{losses:10.8f} \t Accuracy: {(train_corr.item()/(b*100) * 100):10.3f}%")
    
    train_losses.append(losses)
    train_correct.append(train_corr)
    
    
    ## for the validation test loss check
    with torch.no_grad():
        for tb, (X_val_test, y_val_test) in enumerate(val_loader):
            y_valtest_pred = model(X_val_test)
            test_predicted = torch.max(y_valtest_pred,1)[1]
            val_corr += (test_predicted == y_val_test).sum()
        
        val_loss = criterions(y_valtest_pred,y_val_test)
        val_losses.append(val_loss)
        val_correct.append(val_corr)
plt.plot(range(epochs),train_losses,label="Training Loss")
plt.plot(range(epochs),val_losses,label="Validation Loss")
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.plot([(tc.item()/33600)*100 for tc in train_correct], label="Training Accuracy")
plt.plot([(v.item()/8400)*100 for v in val_correct], label="Validation Accuracy")
plt.title("Accuracy plot for Training and Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
sample_submission_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_submission_df.shape ## Note it has got the same length as test dataframe
## making predictions on the test dataset

with torch.no_grad():
    for X_test in test_loader:
        y_test_pred = model(X_test)
        test_predicted = torch.max(y_test_pred,1)[1]

print(f"{test_predicted}")
## Generate the submission file

output = pd.DataFrame({'ImageId':sample_submission_df['ImageId'],'Label':test_predicted})
output.to_csv('mnist_submission_cnn.csv', index=False)