import torch
print(torch.__version__)
# Uncomment the following if you're not running this as a kernel.
# It will download the datasets to an "input" directory. If you're using
# a kernel, the datasets will already be present.

# ! kaggle competitions download -c digit-recognizer -p '../input'
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
print("Number of training & test samples:",
      data_train.shape[0], data_test.shape[0])
# Let's also take a quick look at first few samples.
data_train.head()
# Convert this to a set of X and y.
# I convert the type here for the pixel data to np.uint8 as this makes
# it easier to use some of the torchvision functions later.

X_data = data_train.drop("label", axis=1).values.astype(np.uint8)
y_data = data_train["label"].values

X_test = data_test.values.astype(np.uint8)
def plot_sample_images(X, y, ncols=5, num=25):
    num = int(min(num, len(y)))
    nrows = math.ceil(num / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols, nrows))
    fig.subplots_adjust(hspace=0.8)
    for i, ax in enumerate(axes.flatten()):
        if i < num:
            ax.imshow(X.max()-X[i].reshape(28, 28), cmap="gray")
            ax.set_title(str(y[i]))
        ax.axis('off')
    plt.show()
plot_sample_images(X_data, y_data, ncols=10, num=100)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import PIL.Image

# This will set the hardware device to the GPU if it's available, and
# fall back to using the CPU otherwise.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    num_workers = 1
    pin_memory = True
else:
    device = torch.device("cpu")
    num_workers = 4
    pin_memory = False
print("Using device:", device)
class DigitsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # We reshape the image data into the form
        # expected by torchvision.
        X = np.reshape(self.X[idx], (28, 28, 1))
        if self.transforms is not None:
            X = self.transforms(X)
        if self.y is not None:
            return X, self.y[idx]
        else:
            return X
transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),
                                scale=(0.95, 1.05), shear=10,
                                resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# We don't want to include any random transformations for the test set.
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
train_data = DigitsDataset(X=X_data, y=y_data, transforms=transform_train)
train_data_base = DigitsDataset(X=X_data, y=y_data, transforms=transform_test)
test_data = DigitsDataset(X=X_test, transforms=transform_test)
val_fraction = 0.2 # Use 20% of available training data for validation.

num_input = len(X_data)
val_num = int(num_input * val_fraction)
indices = torch.randperm(num_input)
train_indices = indices[:-val_num]
val_indices = indices[-val_num:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=False, sampler=train_sampler,
                          pin_memory=pin_memory, num_workers=num_workers)
val_loader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=False, sampler=val_sampler,
                        pin_memory=pin_memory, num_workers=num_workers)
val_loader_no_rot = DataLoader(train_data_base, batch_size=batch_size,
                               shuffle=False, sampler=val_sampler,
                               pin_memory=pin_memory, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                         pin_memory=pin_memory, num_workers=num_workers)
# Get a batch of training images. These will include any
# tranformations we have set up on our training data loader,
# allowing us to confirm we have picked reasonable values.
dataiter = iter(train_loader)
images, labels = dataiter.next()

plot_sample_images(images.numpy(), labels.numpy())
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Here we define which functions we'll be using in our model.
        # Those without any learnable parameters, such as dropout, and
        # usually the nonlinear function can be reused in different layers.
        # You need to ensure the various sizes match up here, which can
        # sometimes be tricky, particularly when we go from the conv layers
        # to the fc layers.
        self.nonlin = nn.ReLU()
        self.do_conv = nn.Dropout2d(0.1)
        self.do_fc = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn_c1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn_c2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 500)
        self.bn_fc1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn_fc2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        # The forward function defines how we combine the functions and parameters
        # contained in our model to give us some output from our input data.
        x = self.do_conv(self.pool(self.nonlin(self.bn_c1(self.conv1(x)))))
        x = self.pool(self.nonlin(self.bn_c2(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.do_fc(self.nonlin(self.bn_fc1(self.fc1(x))))
        x = self.do_fc(self.nonlin(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # This is a helper function that will save us needing to hard code the
        # number of features in our forward function above.
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Once we have defined it, we can initialize an instance of our model,
# and transfer it to "device", which will move it to the GPU if it's
# available.
net = Net()
net.to(device)
def evaluate_model(dataloader, model, criterion, optimizer, device):
    # We do this with the model in evaluation mode, which turns off
    # dropout/batchnorm allowing better assessment of the model performance.
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data in dataloader:
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # Calculate the error on the data set
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    average_loss = total_loss / len(dataloader)
    average_error = 100 * (1 - correct / total)
    return average_loss, average_error
def set_optimizer_lr(optimizer, lr):
    # A callback to set the learning rate in an optimizer, without
    # rebuilding the whole optimizer.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr_scale(initial_period, global_idx):
    # This returns a scaling factor which can be used to multiply the
    # learning rate to yield a cosine damping. The period of the cosine
    # is doubled each time the end of the period is reached.
    global_idx = float(global_idx)
    period = initial_period
    while global_idx > period:
        global_idx = global_idx - period
        period = period * 2.
    radians = math.pi * (global_idx / period)
    return 0.5 * (1.0 + math.cos(radians))
criterion = nn.CrossEntropyLoss()
lr = 0.005
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
total_epochs = 0

training_loss_history = []
validation_loss_history = []
training_error_history = []
validation_error_history = []

lr_period = 10.0 * (1.0 - val_fraction)
# I've set this to 8.0 rather than the recommended 10.0 as I used 0.8
# of the training data, so periods should generally match epochs a
# little better with this.
lr_history = []
# This will calculate epochs such that SGDR will end with the lr at a minimum
period = lr_period
max_epochs = 200
suggested_epochs = []
for epoch in range(max_epochs):
    start = len(train_loader) * epoch
    for i in range(len(train_loader)):
        global_step = i + start
        if global_step > period:
            period *= 2
            if epoch > 1:
                suggested_epochs.append(epoch)
print("SGDR Periods will finish on epochs:", suggested_epochs)
# This cell has been constructed such that it can be re-run after it finished
# and SGDR will pick up where it left off. You'll likely want to adjust the
# number of epochs you train for if you do this repeatedly, to match with
# SGDR periods.
for epoch in range(48): # Loop over the dataset multipe times

    net.train()
    start_batch_idx = len(train_loader) * total_epochs
    for batch_idx, data in enumerate(train_loader, total_epochs):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # sgdr adjustment of learning rate
        global_step = batch_idx + start_batch_idx
        batch_lr = lr * sgdr_scale(lr_period, global_step)
        lr_history.append(batch_lr)
        optimizer = set_optimizer_lr(optimizer, batch_lr)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Calculate the the training and validation loss and error rate.
    with torch.no_grad():            
        training_loss, training_error = evaluate_model(train_loader, net,
                                                       criterion, optimizer, device)
        training_error_history.append(training_error)
        training_loss_history.append(training_loss)
        
        validation_loss, validation_error = evaluate_model(val_loader, net,
                                                           criterion, optimizer, device)
        validation_error_history.append(validation_error)
        validation_loss_history.append(validation_loss)

    total_epochs += 1
    print('%3d: training loss & error: %.3f, %.3f%%, validation loss & error: %.3f, %.3f%%' %
          (total_epochs, training_loss,
           training_error, validation_loss, validation_error))
            
print('Finished Training')
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 10))
fig.subplots_adjust(hspace=0)

axs[0].plot(np.linspace(0, total_epochs, num=len(lr_history)), lr_history)
axs[0].set_ylabel("Learning Rate")

axs[1].plot(training_loss_history)
axs[1].plot(validation_loss_history)
axs[1].legend(["Training loss", "Validation loss"])
axs[1].set_ylabel("Loss")

axs[2].plot(training_error_history)
axs[2].plot(validation_error_history)
axs[2].legend(["Training error", "Validation error"])
axs[2].set_ylabel("Average Error (%)")
axs[2].set_xlabel("Epoch")

plt.show()
net.eval()
correct = 0
total = 0
bad_images, bad_labels, bad_pred = [], [], []
with torch.no_grad():
    for data in val_loader_no_rot:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        bad_indices = labels != predicted
        if np.count_nonzero(bad_indices) > 0:
            bad_images.extend(images[bad_indices].cpu().numpy())
            bad_labels.extend(labels[bad_indices].cpu().numpy())
            bad_pred.extend(predicted[bad_indices].cpu().numpy())
        
print('%3d incorrect predictions on the %d validation images: %4.3f %%' % (
    total - correct, total, 100 * (1 - correct / total)))
print('Total training epochs: %d' % (total_epochs))
bad_images = np.array(bad_images)
titles = [str(bad_labels[i]) + " as " + str(bad_pred[i]) for i in range(len(bad_labels))]
plot_sample_images(bad_images, titles, num=len(bad_labels), ncols=10)
nlabels = 10
confusion_mat = np.zeros((nlabels, nlabels))

with torch.no_grad():
    for data in val_loader_no_rot:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted != labels).squeeze()
        for i in range(labels.size()[0]):
            confusion_mat[labels[i], predicted[i]] += c[i].item()
# And let's visualize it also
import itertools

plt.imshow(confusion_mat, cmap=plt.cm.Blues)
plt.colorbar().set_label("Number Incorrect")
plt.title("Confusion Matrix Off-Diagonals")
plt.xticks(range(10))
plt.yticks(range(10))
thresh = confusion_mat.max() / 2.
for i, j in itertools.product(range(nlabels), range(nlabels)):
        if i != j:
            plt.text(j, i, format(int(confusion_mat[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
train_loader_full = DataLoader(train_data, batch_size=batch_size,
                               shuffle=True, pin_memory=pin_memory,
                               num_workers=num_workers)

total_epochs = 0

lr_history = []
# We no longer have a validation set to assess.
training_loss_history = []
training_error_history = []

lr_period = 10.0 # We can set this back to the recommended value.
for epoch in range(12):

    net.train()
    # We've changed the loader named in the following two lines.
    start_batch_idx = len(train_loader_full) * total_epochs
    for batch_idx, data in enumerate(train_loader_full, total_epochs):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # sgdr adjustment of learning rate
        global_step = batch_idx + start_batch_idx
        batch_lr = lr * sgdr_scale(lr_period, global_step)
        lr_history.append(batch_lr)
        optimizer = set_optimizer_lr(optimizer, batch_lr)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Calculate the the training loss and error rate.
    with torch.no_grad():            
        training_loss, training_error = evaluate_model(train_loader, net,
                                                       criterion, optimizer, device)
        training_error_history.append(training_error)
        training_loss_history.append(training_loss)

    total_epochs += 1
    print('%3d: training loss & error: %.3f, %.3f%%' %
          (total_epochs, training_loss, training_error))
            
print('Finished Training')
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 10))
fig.subplots_adjust(hspace=0)

axs[0].plot(np.linspace(0, total_epochs, num=len(lr_history)), lr_history)
axs[0].set_ylabel("Learning Rate")

axs[1].plot(training_loss_history)
axs[1].legend(["Training loss"])
axs[1].set_ylabel("Loss")

axs[2].plot(training_error_history)
axs[2].legend(["Training error"])
axs[2].set_ylabel("Average Error (%)")
axs[2].set_xlabel("Epoch")

plt.show()
net.eval()
test_predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
test_predictions = np.array(test_predictions)
# Add the test index as requested in the submission format description.
test_predictions = np.stack((np.arange(1, len(test_predictions)+1), test_predictions), axis=1)

# Save it to a csv file with the header as requested
np.savetxt("submission.csv", test_predictions, fmt='%d', delimiter=',', header='ImageId,Label', comments='')
# Uncomment this to enter your submission
#! kaggle competitions submit -c digit-recognizer -f submission.csv -m "PyTorch CNN"