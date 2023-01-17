import os

import torch

import torchvision

import tarfile

from torchvision.datasets.utils import download_url
# Dowload the dataset

dataset_url = "http://files.fast.ai/data/cifar10.tgz"

download_url(dataset_url, '.')
# Extract from archive

with tarfile.open('./cifar10.tgz', 'r:gz') as tar:

    tar.extractall(path='./data')
data_dir = './data/cifar10'



print(os.listdir(data_dir))

classes = os.listdir(data_dir + "/train")

print(classes)
airplane_files = os.listdir(data_dir + "/train/airplane")

print('No. of training examples for airplanes:', len(airplane_files))

print(airplane_files[:5])
ship_test_files = os.listdir(data_dir + "/test/ship")

print("No. of test examples for ship:", len(ship_test_files))

print(ship_test_files[:5])
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
img, label = dataset[0]

print(img.shape, label)

img
print(dataset.classes)
import matplotlib.pyplot as plt



def show_example(img, label):

    print('Label: ', dataset.classes[label], "("+str(label)+")")

    plt.imshow(img.permute(1, 2, 0))
show_example(*dataset[0])
show_example(*dataset[1099])
!pip install jovian --upgrade -q
import jovian
jovian.commit()
import numpy as np



def split_indices(n, val_pct=0.1, seed=99):

    # Determine size of validation set

    n_val = int(val_pct*n)

    # Set the random seed (for reproducibility)

    np.random.seed(seed)

    # Create random permutation of 0 to n-1

    idxs = np.random.permutation(n)

    # Pick first n_val indices for validation set

    return idxs[n_val:], idxs[:n_val]
val_pct = 0.2

rand_seed = 42



train_indices, val_indices = split_indices(len(dataset), val_pct, rand_seed)

print(len(train_indices), len(val_indices))

print('Sample validation indices: ', val_indices[:10])
jovian.log_dataset({

    'dataset_url': dataset_url,

    'val_pct': val_pct,

    'rand_seed': rand_seed

})
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataloader import DataLoader



batch_size=400
# Training sampler and data loader

train_sampler = SubsetRandomSampler(train_indices)

train_dl = DataLoader(dataset, 

                      batch_size, 

                      sampler=train_sampler)



# Validation sampler and data loader

val_sampler = SubsetRandomSampler(val_indices)

val_dl = DataLoader(dataset,

                    batch_size, 

                    sampler=val_sampler)
from torchvision.utils import make_grid



def show_batch(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(20, 20))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(images, 20).permute(1, 2, 0))

        break
show_batch(train_dl)
jovian.commit()
import torch.nn as nn

import torch.nn.functional as F
simple_model = nn.Sequential(

    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),

    nn.MaxPool2d(2, 2)

)
for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = simple_model(images)

    print('out.shape:', out.shape)

    break
model = nn.Sequential(

    nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 16 x 16

    

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 8 x 8



    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 4 x 4

    

    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 2 x 2



    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 1 x 1,

    

    nn.Flatten(), # output: bs x 16

    nn.Linear(128, 10) # output: bs x 10 

)
# list(model.modules())

# dir(model)

# print(model.__str__())

# model.__dict__['_modules']

# model._modules

jovian.log_hyperparams({

    'model': model.__str__()

})
model = to_device(model, device)

for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    probs = F.softmax(out, dim=0)

    print(probs[0])

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

valid_dl = DeviceDataLoader(val_dl, device)

to_device(model, device)
jovian.commit()
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
def fit(epochs, model, loss_fn, train_dl, valid_dl, 

        opt_fn=None, lr=None, metric=None):

    train_losses, val_losses, val_metrics = [], [], []

    

    # Instantiate the optimizer

    if opt_fn is None: opt_fn = torch.optim.SGD

    opt = opt_fn(model.parameters(), lr=lr)

    

    for epoch in range(epochs):

        # Training

        model.train()

        for xb,yb in train_dl:

            train_loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)



        # Evaluation

        model.eval()

        result = evaluate(model, loss_fn, valid_dl, metric)

        val_loss, total, val_metric = result

        

        # Record the loss & metric

        train_losses.append(train_loss)

        val_losses.append(val_loss)

        val_metrics.append(val_metric)

        

        # Print progress

        if metric is None:

            print('Epoch [{}/{}], train_loss: {:4f}, val_loss: {:.4f}'

                  .format(epoch+1, epochs, train_loss, val_loss))

        else:

            print('Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}'

                  .format(epoch+1, epochs, train_loss, val_loss, 

                          metric.__name__, val_metric))

    return train_losses, val_losses, val_metrics
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.sum(preds == labels).item() / len(preds)
val_loss, _, val_acc = evaluate(model, F.cross_entropy, 

                                    valid_dl, metric=accuracy)

print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))
num_epochs = 10

opt_fn = torch.optim.Adam

lr = 0.005
jovian.log_hyperparams({

    'num_epochs': num_epochs,

    'opt_fn': opt_fn.__name__,

    'batch_size': batch_size,

    'lr': lr,

})
history = fit(num_epochs, model, F.cross_entropy, 

              train_dl, valid_dl, opt_fn, lr, accuracy)

train_losses, val_losses, val_metrics = history
jovian.log_metrics({

    'train_loss': train_losses[-1],

    'val_loss': val_losses[-1],

    'val_accuracy': val_metrics[-1]

})
def plot_metric(metric_values):

    """Plot metric values in a line graph"""

    plt.plot(metric_values, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');
plot_metric([val_acc] + val_metrics)
def plot_losses(train_losses, val_losses):

    plt.plot(train_losses, '-x')

    plt.plot(val_losses, '-o')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');
plot_losses([None]+train_losses, [val_loss]+val_losses)
jovian.commit()
test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
def predict_image(img, model):

    # Convert to a batch of 1

    xb = to_device(img.unsqueeze(0), device)

    # Get predictions from model

    yb = model(xb)

    # Pick index with highest probability

    _, preds  = torch.max(yb, dim=1)

    # Retrieve the class label

    return dataset.classes[preds[0].item()]
img, label = test_dataset[0]

plt.imshow(img.permute(1, 2, 0))

print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[1002]

plt.imshow(img.permute(1, 2, 0))

print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
img, label = test_dataset[6153]

plt.imshow(img.permute(1, 2, 0))

print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size), device)
test_loss, _, test_acc = evaluate(model, F.cross_entropy, test_loader, metric=accuracy)

print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
jovian.log_metrics({

    'test_loss': test_loss,

    'test_acc': test_acc

})
torch.save(model.state_dict(), 'cifar10-cnn.pth')
model2 = nn.Sequential(

    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 16 x 16

    

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 8 x 8



    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 4 x 4

    

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 2 x 2



    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2, 2), # output: bs x 16 x 1 x 1,

    

    nn.Flatten(), # output: bs x 16

    nn.Linear(16, 10) # output: bs x 10 

)
model2.load_state_dict(torch.load('cifar10-cnn.pth'))
test_loss, _, test_acc = evaluate(model2, F.cross_entropy, test_loader, metric=accuracy)

print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))
jovian.commit()
jovian.commit(outputs=['cifar10-cnn.pth'])