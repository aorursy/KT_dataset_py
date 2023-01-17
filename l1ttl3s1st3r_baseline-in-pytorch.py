# global configurations of the all kernel

config = {

    "train_epoch" : 50,

    "batch_size": 8,

    "show_train_progress": False,

    "show_input": True,

    "use_scheduler": True

}

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_root = '../input/plates/plates/'

print(os.listdir(data_root))
import shutil 

from tqdm import tqdm



train_dir = 'train'

val_dir = 'val'



class_names = ['cleaned', 'dirty']



for dir_name in [train_dir, val_dir]:

    for class_name in class_names:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):

        dest_dir = os.path.join(train_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

        dest_dir = os.path.join(val_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
!ls train
import torch

import numpy as np

import torchvision

import matplotlib.pyplot as plt

import time

import copy



from torchvision import transforms, models



train_transform = transforms.Compose([

    transforms.RandomApply([

        transforms.ColorJitter(

            brightness=0.5,

            contrast=0.5,

            saturation=0.5,

            hue=0.5

        )

    ]),

    transforms.RandomResizedCrop(224),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



train_dataset = torch.utils.data.ConcatDataset([

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform),

    torchvision.datasets.ImageFolder(train_dir, train_transform)

])

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)



batch_size = config['batch_size']

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
len(train_dataloader), len(train_dataset)
if config['show_input']:

    X_batch, y_batch = next(iter(train_dataloader))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);
def show_input(input_tensor, title=''):

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()

    plt.pause(0.001)

        

if config['show_input']:

    counter = 5

    while counter > 0:

        X_batch, y_batch = next(iter(train_dataloader))



        for x_item, y_item in zip(X_batch, y_batch):

            show_input(x_item, title=class_names[y_item])

        counter-=1
def train_model(model, loss, optimizer, scheduler, num_epochs):

    global config

    statistics = {

        'train_loss': [],

        'train_acc': [],

        'val_loss': [],

        'val_acc': []

    }

    

    for epoch in range(num_epochs):

        if config['show_train_progress']:

            print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

                if config['use_scheduler']:

                    scheduler.step()

                model.train()  # Set model to training mode

            else:

                dataloader = val_dataloader

                model.eval()   # Set model to evaluate mode



            running_loss = 0.

            running_acc = 0.



            # Iterate over data.

            for inputs, labels in tqdm(dataloader):

                inputs = inputs.to(device)

                labels = labels.to(device)



                optimizer.zero_grad()



                # forward and backward

                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)

                    loss_value = loss(preds, labels)

                    preds_class = preds.argmax(dim=1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss_value.backward()

                        optimizer.step()



                # statistics

                running_loss += loss_value.item()

                running_acc += (preds_class == labels.data).float().mean()



            epoch_loss = running_loss / len(dataloader)

            epoch_acc = running_acc / len(dataloader)



            # note some statistics for future analysis of learning efficiency

            if phase == 'train':

                statistics['train_loss'].append(epoch_loss)

                statistics['train_acc'].append(epoch_acc.item())

            else:

                statistics['val_loss'].append(epoch_loss)

                statistics['val_acc'].append(epoch_acc.item())

    

            if config['show_train_progress']:

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)



    return (model, statistics)
class PlatesNet(torch.nn.Module):

    def __init__(self):

        super(PlatesNet, self).__init__()

        self.resnet = models.resnet152(pretrained=True, progress=False)



        # Disable grad for all conv layers

        for param in self.resnet.parameters():

            param.requires_grad = False



        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features // 2)

        self.act = torch.nn.LeakyReLU()

        self.fc = torch.nn.Linear(self.resnet.fc.in_features // 2, self.resnet.fc.in_features // 4)

        self.act1 = torch.nn.LeakyReLU()

        self.fc1 = torch.nn.Linear(self.resnet.fc.in_features // 4, 2)

    

    def forward(self, X):

        X = self.resnet(X)

        X = self.act(X)

        X = self.fc(X)

        X = self.act1(X)

        X = self.fc1(X)

        

        return X



model = PlatesNet()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
model, stats = train_model(model, loss, optimizer, scheduler, num_epochs=config['train_epoch'])
stats = pd.DataFrame(stats)



stats.plot(y=['val_loss', 'train_loss'])
stats.head()
stats.plot(y=['val_acc', 'train_acc'])
test_dir = 'test'

shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

    

test_dataset = ImageFolderWithPaths(test_dir, val_transforms)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
model.eval()



test_predictions = []

test_img_paths = []

for inputs, labels, paths in tqdm(test_dataloader):

    inputs = inputs.to(device)

    labels = labels.to(device)

    with torch.set_grad_enabled(False):

        preds = model(inputs)

    test_predictions.append(

        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    test_img_paths.extend(paths)

    

test_predictions = np.concatenate(test_predictions)
if config['show_input']:

    inputs, labels, paths = next(iter(test_dataloader))



    for img, pred in zip(inputs, test_predictions):

        show_input(img, title=pred)
submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')

submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

submission_df.set_index('id', inplace=True)

submission_df.head(n=6)
submission_df.to_csv('submission.csv')
!rm -rf train val test