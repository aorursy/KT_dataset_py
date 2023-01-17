!pip install torchsummary
import os

import time

from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms, utils

from torchsummary import summary

from sklearn.metrics import accuracy_score, confusion_matrix



from PIL import Image
writer = SummaryWriter('runs/fashion-mnist-1')
os.listdir("/kaggle/input")
data_path = os.path.join("/kaggle", "input", "fashionmnist")

train_path = os.path.join(data_path, "fashion-mnist_train.csv")

test_path = os.path.join(data_path, "fashion-mnist_test.csv")

print(train_path, test_path)
train_df = pd.read_csv(train_path) 

train_df.head()
test_df = pd.read_csv(test_path)

test_df.head()
print(train_df.shape, test_df.shape)

print(train_df.info())

print(test_df.info())
train_df.iloc[0, 1:].max()
train_df['label'].value_counts()
test_df['label'].value_counts()
data_path = os.path.join("/kaggle", "working", "fashionmnist")

if not os.path.exists(data_path):

    os.mkdir(data_path)

train_path = os.path.join(data_path, "train")

test_path = os.path.join(data_path, "test")

print(train_path, test_path)
for path in [train_path, test_path]:

    if not os.path.exists(path):

        os.mkdir(path)
class_labels = train_df['label'].unique().tolist()

print(class_labels)
for path in [train_path, test_path]:

    for class_label in class_labels:

        class_path = os.path.join(path, str(class_label))

        if not os.path.exists(class_path):

            os.mkdir(class_path)
for row_id, row in tqdm(train_df.iterrows(), total=len(train_df)):

    label = row['label']

    image_path = os.path.join(data_path, "train", str(label), "{}.png".format(row_id))

    pixels = row.iloc[1:].values.astype("uint8")

    pixels = pixels.reshape(28, 28)

    image = Image.fromarray(pixels)

    image.save(image_path)
for row_id, row in tqdm(test_df.iterrows(), total=len(test_df)):

    label = row['label']

    image_path = os.path.join(data_path, "test", str(label), "{}.png".format(row_id))

    pixels = row.iloc[1:].values.astype("uint8")

    pixels = pixels.reshape(28, 28)

    image = Image.fromarray(pixels)

    image.save(image_path)
data_path = os.path.join("/kaggle", "working", "fashionmnist")

train_path = os.path.join(data_path, "train")

test_path = os.path.join(data_path, "test")

print(train_path, test_path)
mean = 0.5

sd = 0.5

transform = transforms.Compose([

    transforms.Grayscale(),

    transforms.ToTensor(),

    transforms.Normalize((mean, ), (sd, ))

])
train_set = datasets.ImageFolder(root=train_path, transform=transform)

test_set = datasets.ImageFolder(root=test_path, transform=transform)
train_batch_size = 64

test_batch_size = 4
num_cpus = os.cpu_count()

print(num_cpus)
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_cpus) 

test_loader = DataLoader(test_set, batch_size=4, shuffle=test_batch_size, num_workers=num_cpus)
len_train_loader = len(train_loader)

print(len_train_loader)
label_to_name = {

    "0": "t-shirt",

    "1": "trouser",

    "2": "pullover",

    "3": "dress",

    "4": "coat",

    "5": "sandal", 

    "6": "shirt",

    "7": "sneaker",

    "8": "bag",

    "9": "ankle_boot"

}
name_to_label = {v: k for k, v in label_to_name.items()}

print(name_to_label)
def plot_image(image_tensor, title=""):

    """

    Arguments:

    image_tensor -- tensor of Size([1, n_h, n_w]) / Size([3, n_h+4, n_w*m+4])

    """

    # image_tensor.squeeze() -> doesn't work for grids of Size([3, n_h+4, n_c*m+4])

    image_tensor = image_tensor.mean(dim = 0) # Size([28, 28])

    image_tensor = (image_tensor * sd) + mean

    image_tensor = image_tensor * 255 # not necessary

    image_array = image_tensor.numpy()

    plt.figure(figsize=(20, 20))

    plt.imshow(image_array, cmap='Greys')

    plt.title(title)

    return plt.show()
train_iter = iter(train_loader)

X, Y = train_iter.next()

m = min(16, len(train_loader))

image_grid = utils.make_grid(X[:m])

names = [label_to_name[str(label.item())] for label in Y[:m]]

plot_image(image_grid, title=names)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

device = torch.device(device)

print(device)
class Model(nn.Module):

    

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0) # [m, 6, 24, 24]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # [m, 6, 12, 12]

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0) # [m, 16, 8, 8]

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # [m, 16, 4, 4]

        self.fc3 = nn.Linear(in_features=16*4*4, out_features=120) # [m, 120]

        self.fc4 = nn.Linear(in_features=120, out_features=84) # [m, 84]

        self.fc5 = nn.Linear(in_features=84, out_features=10) # [m, 10]

        

    def forward(self, X):

        X = self.pool1(F.relu(self.conv1(X)))

        X = self.pool2(F.relu(self.conv2(X)))

        X = X.view(-1, 16*4*4)

        X = F.relu(self.fc3(X))

        X = F.relu(self.fc4(X))

        X = self.fc5(X) 

        return X
model = Model()

model = nn.DataParallel(model)

model = model.to(device)

print(model)
#list(model.parameters())
summary(model, input_size=(1, 28, 28))
writer.add_graph(model, X)

writer.close()
lr = 0.001

step_size = len(train_loader) * 4

gamma = 0.95

print(step_size)
criterion = nn.CrossEntropyLoss(reduction="mean")

optimizer = optim.Adam(model.parameters(), lr=lr)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
epochs = 20

print_every_n_epochs = 1
epoch_losses = []

epoch_lrs = []

iteration_losses = []

iteration_lrs = []



for epoch in tqdm(range(1, epochs+1)):

    epoch_loss = 0

    epoch_lr = 0

    

    for X, Y in tqdm(train_loader, desc="Epoch-{}".format(epoch)):

        X, Y = X.to(device), Y.to(device)

        

        optimizer.zero_grad()

        Y_pred_logits = model(X)

        loss = criterion(Y_pred_logits, Y)

        loss.backward()

        optimizer.step()

        lr_scheduler.step()

        

        iteration_losses.append(loss.item())

        iteration_lrs.append(lr_scheduler.get_lr()[0])

        epoch_loss += loss.item()

        epoch_lr += lr_scheduler.get_lr()[0]

        

    epoch_loss /= len(train_loader)

    epoch_lr /= len(train_loader)

    epoch_losses.append(epoch_loss)

    epoch_lrs.append(epoch_lr)

    

    if epoch % print_every_n_epochs == 0:    

        message = "Epoch:{}    Loss:{}    LR:{}".format(epoch, epoch_loss, epoch_lr)

        print(message)

        
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1.plot(epoch_losses, marker="o", markersize=5)

ax1.set_title("Loss")

ax2.plot(epoch_lrs, marker="o", markersize=5)

ax2.set_title("LR")

plt.xlabel("Epochs")

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

ax1.plot(iteration_losses[::100])

ax1.set_title("Loss")

ax2.plot(iteration_lrs[::100])

ax2.set_title("LR")

plt.xlabel("Iterations")

plt.show()
window = 100

pd.Series(iteration_losses).rolling(window=window).mean().iloc[window-1:].plot()

plt.show()
working_path = os.path.join("/kaggle", "working", "fashionmnist")

path = os.path.join(working_path, "classifier.pth")

torch.save(model.state_dict(), path)
working_path = os.path.join("/kaggle", "working", "fashionmnist")

path = os.path.join(working_path, "classifier.pth")

model = Model()

model = nn.DataParallel(model)

model.load_state_dict(torch.load(path))

#model = model.to("cpu")
for p in model.parameters():

    print(p.is_cuda)
with torch.no_grad():

    Y_train, Y_pred_train = [], []

    for X_mb, Y_mb in tqdm(train_loader):

        out = model(X_mb)

        _, Y_pred_mb = torch.max(out, 1)

        Y_train.extend(Y_mb.numpy().tolist())

        Y_pred_train.extend(Y_pred_mb.cpu().numpy().tolist())
with torch.no_grad():

    Y_test, Y_pred_test = [], []

    for X_mb, Y_mb in tqdm(test_loader):

        out = model(X_mb)

        _, Y_pred_mb = torch.max(out, 1)

        Y_test.extend(Y_mb.numpy().tolist())

        Y_pred_test.extend(Y_pred_mb.cpu().numpy().tolist())
train_accuracy = accuracy_score(Y_train, Y_pred_train)

test_accuracy = accuracy_score(Y_test, Y_pred_test)

print("Train Accuracy: {}".format(train_accuracy))

print("Test Accuracy: {}".format(test_accuracy))
labels = [label_to_name[str(i)] for i in range(10)]

c_mat_train = confusion_matrix(Y_train, Y_pred_train)

c_mat_test = confusion_matrix(Y_test, Y_pred_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.heatmap(c_mat_train, annot=True, fmt='d', ax=ax1, xticklabels=labels, yticklabels=labels)

ax1.set_title('Train Data')

ax1.set_xlabel('Predicted')

ax1.set_ylabel('Actual')

sns.heatmap(c_mat_test, annot=True, fmt='d', ax=ax2, xticklabels=labels, yticklabels=labels)

ax2.set_title('Test Data')

ax2.set_xlabel('Predicted')

ax2.set_ylabel('Actual')

plt.show()
def compute_accuracies(c_mat):

    accuracies = c_mat.astype('float') / c_mat.sum(axis=1)

    accuracies = accuracies.diagonal()

    accuracies = {k:v for k, v in zip(labels, accuracies)}

    return accuracies
compute_accuracies(c_mat_train)
compute_accuracies(c_mat_test)