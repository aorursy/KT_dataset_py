# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torchvision

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

from torchvision.transforms import ToTensor

from torchvision.utils import make_grid

from torch.utils.data import DataLoader, TensorDataset, random_split

%matplotlib inline
sat = pd.read_csv('/kaggle/input/satellite-images/dataset_186_satimage.csv')

sat.head()
classes = sat['class'].unique()

classes
input_cols = sat.iloc[:,:-1].columns

input_cols
input_cols.isnull()
output_cols = sat.iloc[:,-1:].columns

output_cols
label_dict = {}

for label in classes:

    for label in sat['class']:

        if label in label_dict:

            label_dict[label] += 1

        else:

            label_dict[label] = 1

        

label_dict
# Write your answer here

import seaborn as sns

sns.set_style("darkgrid")

class_distribution = sat[['class']]

print(class_distribution.describe())



#plot a distribution of the charges

plt.title("Distribution of Classes")



sns.distplot(class_distribution, kde=False);
categorical_cols = sat.select_dtypes(include=object).columns

categorical_cols
inputs_array = np.array(sat[input_cols])

targets_array = np.array(sat[output_cols])

inputs_array, targets_array
inputs = torch.tensor(inputs_array, dtype=torch.float32)

targets = torch.tensor(targets_array, dtype= torch.long)

targets = targets.squeeze()

targets.size()

inputs.shape
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='Classification using satellite data')
len(sat)
targets.size()
inputs.dtype, targets.dtype
dataset = TensorDataset(inputs, targets)
torch.manual_seed(43)

val_percent = 0.2# between 0.1 and 0.2

val_size = int(sat.shape[0] * val_percent)

train_size = sat.shape[0] - (val_size + 500)

test_size = 500





train_ds, val_ds,test_ds = random_split(dataset, [train_size, val_size,test_size])

 # Use the random_split function to split dataset into 2 parts of the desired length
len(train_ds),len(val_ds)
batch_size = 200
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)
input_size = 36
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class SatelliteModel(nn.Module):

    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self):

        super().__init__()

        # hidden layer

        self.linear1 = nn.Linear(36, 400)

        # hidden layer 2

        self.linear2 = nn.Linear(400, 1200)

        # output layer

        self.linear3 = nn.Linear(1200, 8)

        

    def forward(self, xb):

        # Flatten the image tensors

        out = xb.view(xb.size(0), -1)

        # Get intermediate outputs using hidden layer

        out = self.linear1(out)

        # Apply activation function

        out = F.relu(out)

        # Get intermediate outputs using hidden layer

        out = self.linear2(out)

        # Apply activation function

        out = F.relu(out)

        # Get predictions using output layer

        out = self.linear3(out)

        return out

    

    def training_step(self, batch):

        inputs, targets = batch 

        out = self(inputs)                  # Generate predictions

        loss = F.cross_entropy(out, targets) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        inputs, targets = batch 

        out = self(inputs)                    # Generate predictions

        loss = F.cross_entropy(out, targets)   # Calculate loss

        acc = accuracy(out, targets)           # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
model = SatelliteModel()
for t in model.parameters():

    print(t.shape)
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
for inputs,targets in train_loader:

    print(inputs.shape)

    inputs = to_device(inputs, device)

    print(inputs.device)

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
train_loader = DeviceDataLoader(train_loader, device)

val_loader = DeviceDataLoader(val_loader, device)
for xb, yb in val_loader:

    print('xb.device:', xb.device)

    print('yb:', yb)

    print(yb.shape)

    break
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
# Model (on GPU)

model = SatelliteModel()

to_device(model, device)
history = [evaluate(model, val_loader)]

history
history += fit(5, 0.5, model, train_loader, val_loader)
history += fit(5, 0.1, model, train_loader, val_loader)
history += fit(5, 0.01, model, train_loader, val_loader)
history += fit(10, 0.001, model, train_loader, val_loader)
losses = [x['val_loss'] for x in history]

plt.plot(losses, '-x')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.title('Loss vs. No. of epochs');
accuracies = [x['val_acc'] for x in history]

plt.plot(accuracies, '-x')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Accuracy vs. No. of epochs');
arch = "3 layers (400,1200,8)"
lrs = [0.5, 0.1, 0.01,0.001]
epochs =[5,5,5,10]
jovian.log_hyperparams(arch=arch, 

                       lrs=lrs, 

                       epochs=epochs)
val_loss = 0.2472

val_acc = 0.9090
jovian.log_metrics(val_loss=val_loss, val_acc=val_acc)
def predict_single(input,model):

    xb = to_device(input.unsqueeze(0), device)

    yb = model(xb)

    _, preds  = torch.max(yb, dim=1)

    return preds[0].item()
classes = [0,1,2,3,4,5,6,7]
for n in range(20):

    input, target = test_ds[n]

    print('Actual_Class:', target, ', Predicted:', classes[predict_single(input, model)])
jovian.commit(project ='Classification using satellite data')