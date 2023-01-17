import numpy as np

import pandas as pd

import pandas_profiling

import os

from PIL import Image

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset

import torchvision

import torchvision.transforms as transforms



import matplotlib.pyplot as plt

%matplotlib inline
test_img = Image.open('/kaggle/input/plant-seedlings-classification/train/Maize/92c06eaca.png')

test_img = test_img.resize((200, 200))

test_img = np.array(test_img)

plt.imshow(test_img)

plt.show()

print(test_img.shape)
train_data_dir = '/kaggle/input/plant-seedlings-classification/train'

test_data_dir = '/kaggle/input/plant-seedlings-classification/test'
def get_bad_images(train_path, test_path):

    bad_images = []

    images = []

    

    classes = [dI for dI in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, dI))]

    for cl in classes:

        class_dir = os.path.join(train_path, cl)

        for img in os.listdir(class_dir):

            images.append(os.path.join(class_dir, img))



    for test_img in os.listdir(test_path):

        images.append(os.path.join(test_path, test_img))

            

    all_images = len(images)

    

    for img in images:

        try:

            _ = Image.open(img)

        except:

            bad_images.append(img)

    

    return set(bad_images)

    

bad_images = get_bad_images(train_data_dir, test_data_dir)

bad_images
folders = [dI for dI in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir,dI))]



class_id = {}

id_class = {}

for cl in range(len(folders)):

    class_id[folders[cl]] = cl

    id_class[cl] = [folders[cl]]
len(class_id)
class_id
id_class
train_data = []

val_data = []

for c in range(len(class_id.keys())):

    cl = list(class_id.keys())[c]

    class_path = os.path.join(train_data_dir, cl)

    class_images = [os.path.join(class_path, dI) for dI in os.listdir(class_path)]

    train, validation = train_test_split(class_images, shuffle=True, test_size=0.3)

    

    for t in train:

        train_data.append([t, c])

        

    for v in validation:

        val_data.append([v, c])
len(train_data)
class SeedlingsTrainDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        try:

            img_path, c = self.data[index]

            img = Image.open(img_path)



            if self.transform is not None:

                img = self.transform(img)

        

            return img, c     

        except:

            print('Bad image')

            while True:

                try:

                    index = random.randint(0, len(self.data) - 2)

                    img_path, c = self.data[index]

                    img = Image.open(img_path)



                    if self.transform is not None:

                        img = self.transform(img)

        

                    return img, c

                except:

                    print('One more bad image')
class SeedlingsValidationDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        img_path, c = self.data[index]

        img = Image.open(img_path)

        

        if self.transform is not None:

            img = self.transform(img)

        

        return img, c       
class SeedlingsTestDataset(Dataset):

    def __init__(self, data_path, transform=None):

        self.data_path = data_path

        self.transform = transform

        

        self.images = [dI for dI in os.listdir(self.data_path)]

    

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, index):

        img_path = os.path.join(self.data_path, self.images[index])

        img = Image.open(img_path)

        

        if self.transform is not None:

            img = self.transform(img)

        

        return img, self.images[index]
train_transforms = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),

        transforms.RandomAffine(degrees=180, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),

        transforms.ToTensor(),

        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])
validation_transforms = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.ToTensor(),

        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])
train_dataset = SeedlingsTrainDataset(

    train_data, 

    transform=train_transforms)
img, cl = train_dataset.__getitem__(150)

plt.figure(figsize=(8, 8))

transforms.ToPILImage()(img)
validation_dataset = SeedlingsValidationDataset(

    train_data, 

    transform=validation_transforms)
test_dataset = SeedlingsTestDataset(

    test_data_dir, 

    transform=validation_transforms)
img = test_dataset.__getitem__(150)[0]

print(img.size())

plt.figure(figsize=(8, 8))

transforms.ToPILImage()(img)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)

validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
net = torch.hub.load('pytorch/vision:v0.4.2', 'squeezenet1_1', pretrained=True)
net
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# I used 10 epochs

for epoch in range(0):



    running_loss = 0.0

    for i, data in enumerate(iter(train_loader), 0):

        inputs, labels = data



        optimizer.zero_grad()



        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        if i % 50 == 49:

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 50))

            running_loss = 0.0

    

    scheduler.step()



print('Finished Training')
torch.save(net.state_dict(), '/kaggle/working/squeezenet.pth')
net.load_state_dict(torch.load('/kaggle/working/squeezenet.pth'))
# correct = 0

# top3 = 0

# total = 0

# with torch.no_grad():

#     #batch_size = 1

#     for i in range(len(validation_dataset)):

#         try:

#             images, label = validation_dataset.__getitem__(i)

#             images = images[None, :, :]

            

#             predictions = net(images).numpy()[0].argsort()[::-1]

            

#             if label == predictions[0]:

#                 correct += 1

            

#             if label in predictions[:3]:

#                 top3 += 1

            

#             total += 1

#         except:

#             print('exception')



# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# print('Accuracy of the network on the test images (top 3): %d %%' % (100 * top3 / total))

# print('Total: ', total)
files = []

predictions = []

with torch.no_grad():

    for img, file in test_loader:

        prediction = net(img).numpy()[0].argsort()[-1]

        files.append(file[0])

        predictions.append(id_class[prediction][0])
df = pd.DataFrame({

    'file': files,

    'species': predictions

})
df.shape
df.to_csv('submission.csv', index=False)