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
!pip install jovian --upgrade
import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torchvision.transforms as T

import sklearn.metrics as m

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

from sklearn.model_selection import train_test_split

import jovian

from torchvision.datasets import ImageFolder



%matplotlib inline
project_name = 'Course Project - Blindness Detection'

directory = '../input/aptos2019-blindness-detection'



train_trans = T.Compose([

#     T.RandomCrop(512, padding=8, padding_mode='reflect'),

    T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

#    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

#     T.RandomHorizontalFlip(), 

#     T.RandomRotation(10),

    T.Resize([256,256]),

    T.ToTensor(), 

#     T.Normalize(*imagenet_stats,inplace=True), 

#     T.RandomErasing(inplace=True)

])



valid_trans = T.Compose([

    T.Resize([256,256]),

    T.ToTensor()

])
train_labels = pd.read_csv(directory + '/train.csv' )

labels = {0 : 'No DR',1 : 'Mild', 2 : 'Moderate',3 : 'Severe',4 : 'Proliferative DR'}
plt.imshow(train_trans(Image.open(directory + '/train_images/' + train_labels.id_code.loc[0] + '.png')).permute(1,2,0))
for i in range(10):

    print(train_trans(Image.open(directory + '/train_images/' + train_labels.id_code.loc[i] + '.png')).shape)
train_labels.diagnosis.hist()

plt.xticks([0,1,2,3,4])

plt.grid(False)

plt.show()
def encode_label(label):

    target = torch.zeros(5)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target

class Blindness(Dataset):

    def __init__(self, df, root_dir, transform=None):

        self.df = df

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['id_code'], row['diagnosis']

        img_fname = self.root_dir + "/" + img_id + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img,encode_label(img_label)
train_labels.diagnosis.value_counts()
np.random.seed(40)

cols = ['id_code','diagnosis']

train_len = np.random.randn(len(train_labels)) < 0.75

t_ds,test_ds = train_labels[train_len].reset_index(),train_labels[~train_len].reset_index()

t_ds, test_ds = t_ds[cols],test_ds[cols]



np.random.seed(40)

val_len = np.random.rand(len(t_ds)) < 0.8

train_ds,valid_ds = t_ds[val_len].reset_index(),t_ds[~val_len].reset_index()

train_ds,valid_ds = train_ds[cols],valid_ds[cols]





print("train : {}\ntest : {}\ntrain_split : {}\nvalid_split : {}\n{}".format(t_ds.shape[0],test_ds.shape[0],train_ds.shape[0],valid_ds.shape[0], t_ds.shape[0] == (train_ds.shape[0] + valid_ds.shape[0])))
train_transformed = Blindness(train_ds,directory + '/train_images', transform = train_trans)

valid_transformed = Blindness(valid_ds,directory + '/train_images', transform = valid_trans)

test_transformed = Blindness(test_ds,directory + '/train_images', transform = train_trans)



batch_size = 5

train_dl = DataLoader(train_transformed,batch_size,shuffle = True, num_workers = 4, pin_memory = True)

valid_dl = DataLoader(valid_transformed,batch_size * 2,num_workers = 4, pin_memory = True)
def batch_display(x_dl):

    for i,j in x_dl:

        fig,ax = plt.subplots(figsize = (20,10))

        ax.imshow(make_grid(i,nrow = 10).permute(1,2,0))

        break

batch_display(train_dl)
def F_score(output, label, threshold=0.5, beta=1):

    prob = output > threshold

    label = label > threshold



    TP = (prob & label).sum(1).float()

    TN = ((~prob) & (~label)).sum(1).float()

    FP = (prob & (~label)).sum(1).float()

    FN = ((~prob) & label).sum(1).float()



    precision = torch.mean(TP / (TP + FP + 1e-12))

    recall = torch.mean(TP / (TP + FN + 1e-12))

    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return F2.mean(0)



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.binary_cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.binary_cross_entropy(out, labels)   # Calculate loss

        acc = F_score(out, labels)           # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class Blindness(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        self.network = models.resnet50(pretrained = True)

        f = self.network.fc.in_features

        self.network.fc = nn.Linear(f,5)

#         self.network = nn.Sequential(

#                         nn.Conv2d(3,32,kernel_size = 3, padding = 1), #32 * 256 * 256

#                         nn.ReLU(),

#                         nn.Conv2d(32,64,kernel_size = 3, padding = 1), #64 * 256 * 256

#                         nn.ReLU(),

#                         nn.MaxPool2d(2,2), #64 * 128 * 128

                        

# #                         nn.Conv2d(64,128,kernel_size = 3, padding = 1), #128 * 128 * 128

# #                         nn.ReLU(),

# #                         nn.Conv2d(128,256,kernel_size = 3, padding = 1), #256 * 128 * 128

# #                         nn.ReLU(),

# #                         nn.MaxPool2d(2,2), #256 * 64* 64

            

# #                         nn.Conv2d(256,512,kernel_size = 3, padding = 1), #512 * 64 *64 

# #                         nn.ReLU(),

# #                         nn.Conv2d(512,512,kernel_size = 3, padding = 1), #512 * 64 *64 

# #                         nn.ReLU(),

# #                         nn.MaxPool2d(2,2), #512 * 32 *32 

                        

# #                         nn.Conv2d(512,512,kernel_size = 3, padding = 1), #512 * 32 *32 

# #                         nn.ReLU(),

# #                         nn.Conv2d(512,1024,kernel_size = 3, padding = 1), #1024 * 32 *32 

# #                         nn.ReLU(),

# #                         nn.MaxPool2d(2,2), #1024 * 16 * 16

                        

# #                         nn.Conv2d(1024,1024,kernel_size = 3, padding = 1), #1024 * 16 * 16

# #                         nn.ReLU(),

# #                         nn.Conv2d(1024,1024,kernel_size = 3, padding = 1), #1024 * 16 * 16

# #                         nn.ReLU(),

# #                         nn.MaxPool2d(2,2), #1024 * 8 * 8

            

#                         nn.Flatten(),

#                         nn.Linear(64*32*32,1024),

#                         nn.Linear(1024,512),

#                         nn.Linear(512,5)

#         )

    def forward(self,xb):

        return torch.sigmoid(self.network(xb))

        
model = Blindness()

torch.cuda.empty_cache()

model
for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
F.binary_cross_entropy(out[0],labels[0])
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

val_dl = DeviceDataLoader(valid_dl, device)

to_device(model, device);
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

        for batch in tqdm(train_loader):

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
model = to_device(Blindness(), device)
evaluate(model,val_dl)
num_epochs = 10

opt_func = torch.optim.Adam

lr = 0.001
history = fit(num_epochs,lr,model, train_dl, val_dl, opt_func)
jovian.log_metrics(train_loss=history[-1]['train_loss'], 

                   val_loss=history[-1]['val_loss'], 

                   val_acc=history[-1]['val_acc'])
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

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
def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
def predict_image(img, model):

    # Convert to a batch of 1

    xb = to_device(img.unsqueeze(0), device)

    # Get predictions from model

    yb = model(xb)

    # Pick index with highest probability

    _, preds  = torch.max(yb, dim=1)

    # Retrieve the class label

    return labels[preds[0].item()]
img, label = test_transformed[0]

plt.imshow(img.permute(1, 2, 0))

print('Label:', label ,', Predicted:', predict_image(img, model))
img, label = test_transformed[120]

plt.imshow(img.permute(1, 2, 0))

print('Label:', label,', Predicted:', predict_image(img, model))
test_loader = DeviceDataLoader(DataLoader(test_transformed, batch_size*2), device)

result = evaluate(model, test_loader)

result
jovian.log_metrics(test_loss=result['val_loss'], test_acc=result['val_acc'])
@torch.no_grad()

def predict_dl(dl, model):

    torch.cuda.empty_cache()

    batch_probs = []

    for xb, _ in tqdm(dl):

        probs = model(xb)

        batch_probs.append(probs.cpu().detach())

    batch_probs = torch.cat(batch_probs)

    return [decode_target(x) for x in batch_probs]
test_preds = predict_dl(test_loader, model)

submission_df = pd.read_csv(directory + '/test.csv') 

submission_df.Label = test_preds
sub_fname = 'submission.csv'

submission_df.to_csv(sub_fname, index=False)
jovian.commit(project = project_name)