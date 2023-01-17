import torch

import numpy as np

import torchvision

from torchvision.datasets import MNIST

from torchvision.transforms import ToTensor

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataloader import DataLoader
dataset = MNIST(root='data/', 

                download=True, 

                transform=ToTensor())
def split_indices(n, val_pct):

    # Determine size of validation set

    n_val = int(val_pct*n)

    # Create random permutation of 0 to n-1

    idxs = np.random.permutation(n)

    # Pick first n_val indices for validation set

    return idxs[n_val:], idxs[:n_val]
train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)



print(len(train_indices), len(val_indices))

print('Sample val indices: ', val_indices[:20])
batch_size=100



# Training sampler and data loader

train_sampler = SubsetRandomSampler(train_indices)

train_dl = DataLoader(dataset, 

                      batch_size, 

                      sampler=train_sampler)



# Validation sampler and data loader

valid_sampler = SubsetRandomSampler(val_indices)

valid_dl = DataLoader(dataset,

                    batch_size, 

                    sampler=valid_sampler)
import torch.nn.functional as F

import torch.nn as nn
class MnistModel(nn.Module):

    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, hidden_size, out_size):

        super().__init__()

        # hidden layer

        self.linear1 = nn.Linear(in_size, hidden_size)

        # output layer

        self.linear2 = nn.Linear(hidden_size, out_size)

        

    def forward(self, xb):

        # Flatten the image tensors

        xb = xb.view(xb.size(0), -1)

        # Get intermediate outputs using hidden layer

        out = self.linear1(xb)

        # Apply activation function

        out = F.relu(out)

        # Get predictions using output layer

        out = self.linear2(out)

        return out
input_size = 784

num_classes = 10



model = MnistModel(input_size, hidden_size=32, 

                   out_size=num_classes)
for t in model.parameters():

    print(t.shape)
for images, labels in train_dl:

    outputs = model(images)

    loss = F.cross_entropy(outputs, labels)

    print('Loss:', loss.item())

    break



print('outputs.shape : ', outputs.shape)

print('Sample outputs :\n', outputs[:2].data)
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
for images, labels in train_dl:

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
train_dl = DeviceDataLoader(train_dl, device)

valid_dl = DeviceDataLoader(valid_dl, device)
for xb, yb in valid_dl:

    print('xb.device:', xb.device)

    print('yb:', yb)

    break
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):

    # Generate predictions

    preds = model(xb)

    # Calculate loss

    loss = loss_func(preds, yb)

                     

    if opt is not None:

        # Compute gradients

        loss.backward()

        # Update parameters             

        opt.step()

        # Reset gradients

        opt.zero_grad()

    

    metric_result = None

    if metric is not None:

        # Compute the metric

        metric_result = metric(preds, yb)

    

    return loss.item(), len(xb), metric_result
def evaluate(model, loss_fn, valid_dl, metric=None):

    with torch.no_grad():

        # Pass each batch through the model

        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)

                   for xb,yb in valid_dl]

        # Separate losses, counts and metrics

        losses, nums, metrics = zip(*results)

        # Total size of the dataset

        total = np.sum(nums)

        # Avg. loss across batches 

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None

        if metric is not None:

            # Avg. of metric across batches

            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric
def fit(epochs, lr, model, loss_fn, train_dl, 

        valid_dl, metric=None, opt_fn=None):

    losses, metrics = [], []

    

    # Instantiate the optimizer

    if opt_fn is None: opt_fn = torch.optim.SGD

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    

    for epoch in range(epochs):

        # Training

        for xb,yb in train_dl:

            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)



        # Evaluation

        result = evaluate(model, loss_fn, valid_dl, metric)

        val_loss, total, val_metric = result

        

        # Record the loss & metric

        losses.append(val_loss)

        metrics.append(val_metric)

        

        # Print progress

        if metric is None:

            print('Epoch [{}/{}], Loss: {:.4f}'

                  .format(epoch+1, epochs, val_loss))

        else:

            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'

                  .format(epoch+1, epochs, val_loss, 

                          metric.__name__, val_metric))

    return losses, metrics
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.sum(preds == labels).item() / len(preds)
# Model (on GPU)

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

to_device(model, device)
val_loss, total, val_acc = evaluate(model, F.cross_entropy, 

                                    valid_dl, metric=accuracy)

print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))
losses1, metrics1 = fit(5, 0.5, model, F.cross_entropy, 

                        train_dl, valid_dl, accuracy)
losses2, metrics2 = fit(5, 0.1, model, F.cross_entropy, 

                        train_dl, valid_dl, accuracy)
import matplotlib.pyplot as plt
# Replace these values with your results

accuracies = [val_acc] + metrics1 + metrics2

plt.plot(accuracies, '-x')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Accuracy vs. No. of epochs');
!pip install jovian --upgrade -q
import jovian
jovian.commit()