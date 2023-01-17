import os

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from PIL import Image

from torch.utils.data import Dataset

from torch.utils.data import random_split

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid

from torchvision.transforms import ToTensor

from sklearn.metrics import f1_score
data_folder = '/kaggle/input/jovian-pytorch-z2g/Human protein atlas/'

working_folder = '/kaggle/working/'

train_folder = data_folder + 'train/'

test_folder = data_folder + 'test/'
train_labels = pd.read_csv(data_folder + 'train.csv', index_col='Image')

train_labels = train_labels['Label'].str.get_dummies(sep=" ").sort_values(by='Image') # One-hot encoding

train_labels.head()
len(train_labels)
train_labels.sum()
plt.hist(train_labels.sum(axis=1))
image_name = str(train_labels.sample(1).index[0]) + '.png'

image_path = train_folder + image_name

plt.axis('off')

plt.imshow(Image.open(image_path))
class ProteinAtlas(Dataset):

    

    classes = ['Mitochondria',

               'Nuclear bodies',

               'Nucleoli',

               'Golgi apparatus',

               'Nucleoplasm',

               'Nucleoli fibrillar center',

               'Cytosol',

               'Plasma membrane',

               'Centrosome',

               'Nuclear speckles']

    

    def __init__(self, folder_path, csv_path='', transform=None):

        self.path = folder_path

        self.transform = transform

        

        if csv_path:

            data = pd.read_csv(csv_path, index_col='Image')

            self.data = data['Label'].astype(str).str.get_dummies(sep=" ").sort_values(by='Image') # One-hot encoding

        else:

            # If no CSV file, inspect folder

            index_array = []

            for filename in os.listdir(folder_path):

                index_array.append(int(filename[:-4]))

            self.data = pd.DataFrame(0, index=index_array, columns=range(10))



    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        filepath = self.path + str(self.data.iloc[idx].name) + '.png'

        image = Image.open(filepath)

        labels = torch.tensor(self.data.iloc[idx].tolist(), dtype=torch.float)



        if self.transform:

            image = self.transform(image)

                    

        sample = (image, labels)

            

        return sample
labeled_dataset = ProteinAtlas(train_folder, data_folder + 'train.csv', transform=ToTensor())



val_size = 4000

train_size = len(labeled_dataset) - val_size



torch.manual_seed(442)

train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size])
batch_size = 64



val_dl = DataLoader(val_dataset, batch_size, num_workers=4, pin_memory=True)

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
images, labels = next(iter(train_dl))

fig, ax = plt.subplots(figsize=(20,20))

ax.set_xticks([]); ax.set_yticks([])

ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
def accuracy(outputs, labels, threshold=0.5):

    # Macro F-score

    # The F-score of each class is computed, whose sum is the returned as the accuracy of a batch

    return f1_score(labels.data.cpu(), (outputs > threshold).data.cpu(), average='macro')





class ProteinClassificationModel(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # AlexNet architecture

            nn.Conv2d(3, 96, kernel_size=11, stride=4),

            nn.ReLU(),                       # output: (126 x 126 x 96)

            nn.MaxPool2d((3, 3), stride=2),  # output: (62 x 62 x 96)



            nn.Conv2d(96, 256, kernel_size=5, padding=2),

            nn.ReLU(),                       # output: (62 x 62 x 256)

            nn.MaxPool2d((3, 3), stride=2),  # output: (30 x 30 x 256)



            nn.Conv2d(256, 384, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(),                       # output: (30 x 30 x 256)

            nn.MaxPool2d((3, 3), stride=2),  # output: (14 x 14 x 256)



            nn.Flatten(),

            nn.Linear(14 * 14 * 256, 4096),

            nn.ReLU(),

            nn.Linear(4096, 4096),

            nn.ReLU(),

            nn.Linear(4096, 10),

            nn.Sigmoid() # Multi-label prediction

        )

        

        self.loss_f = F.binary_cross_entropy

        

    def forward(self, xb):

        return self.network(xb)

    

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = self.loss_f(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch

        out = self(images)                    # Generate predictions

        loss = self.loss_f(out, labels)   # Calculate loss

        acc = accuracy(out, labels)  # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': torch.tensor(acc)}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

        

model = ProteinClassificationModel()
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

print(device)



train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device)
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
evaluate(model, val_dl)
num_epochs = 10

opt_func = torch.optim.Adam

lr = 0.001



history = []
history += fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
torch.save(model.state_dict(), working_folder + 'model.pth')
pred_model = to_device(ProteinClassificationModel(), device)

pred_model.load_state_dict(torch.load('/kaggle/input/model1/model.pth'))
test_dataset = ProteinAtlas(test_folder, '/kaggle/input/jovian-pytorch-z2g/submission.csv', transform=ToTensor())

test_dl = DeviceDataLoader(DataLoader(test_dataset, 128, num_workers=4, pin_memory=True), device)



results = []

threshold = 0.5



for images, _ in test_dl:

    pred_labels = pred_model(images) > threshold

    results += [' '.join(str(i) for i, value in enumerate(label_vector) if value) for label_vector in pred_labels]

    

print(results)
submission = pd.read_csv('/kaggle/input/jovian-pytorch-z2g/submission.csv', index_col='Image')

submission['Label'] = results

submission.to_csv('/kaggle/working/submission.csv')