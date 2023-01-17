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

        if i % 6 != 0:

            dest_dir = os.path.join(train_dir, class_name) 

        else:

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

train_transforms = transforms.Compose([

    torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),

    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),

    transforms.RandomRotation(40, resample=False, expand=False, center=None),

    transforms.RandomVerticalFlip(p=0.5),

#     transforms.RandomResizedCrop(224),

    transforms.CenterCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)



batch_size = 10

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
len(train_dataloader), len(train_dataset)
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



X_batch, y_batch = next(iter(train_dataloader))



for x_item, y_item in zip(X_batch, y_batch):

    show_input(x_item, title=class_names[y_item])
test_accuracy_history = []

test_loss_history = []

val_accuracy_history = []

val_loss_history = []

def train_model(model, loss, optimizer, scheduler, num_epochs):

    for epoch in range(num_epochs):

        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

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

            ##

            if (phase=='train'):

                test_accuracy_history.append(epoch_acc)

                test_loss_history.append(epoch_loss)

            elif (phase=='val'):

                val_accuracy_history.append(epoch_acc)

                val_loss_history.append(epoch_loss)

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)



    return model
model = models.resnet18(pretrained=True)



# Disable grad for all conv layers

for param in model.parameters():

#     print(param)

    t = param

    param.requires_grad = False





# model.fc = torch.nn.Linear(model.fc.in_features, 100)

# # model.bn = torch.nn.BatchNorm1d(100)

# model.act  = torch.nn.ReLU()



# model.fc1 = torch.nn.Linear(100, 100)

# # model.bn1 = torch.nn.BatchNorm1d(100)

# model.act1  = torch.nn.ReLU()



# model.fc2 = torch.nn.Linear(100, 50)

# # model.bn2 = torch.nn.BatchNorm1d(50)

# model.act2  = torch.nn.ReLU()



# model.bn3 = torch.nn.BatchNorm1d(50)

# model.fc3 = torch.nn.Linear(50, 2) 

    

    

    

num_ftrs = model.fc.in_features

# model.bn = torch.nn.BatchNorm1d(num_ftrs)

model.fc = torch.nn.Linear(num_ftrs, 2)



num_ftrs = model.fc.in_features

model.bn = torch.nn.BatchNorm1d(num_ftrs)

model.fc = torch.nn.Linear(num_ftrs, 2)    

    

model.act = torch.nn.ReLU()

model.fc = torch.nn.Linear(model.fc.in_features, model.fc.in_features)

model.act2 = torch.nn.ReLU()



model.dd = torch.nn.Dropout(p=0.5)



model.act7 = torch.nn.ReLU()

model.fc8 = torch.nn.Linear(model.fc.in_features, model.fc.in_features)

model.act9 = torch.nn.ReLU()



model.dd10 = torch.nn.Dropout(p=0.5)



model.fc2 = torch.nn.Linear(model.fc.in_features, model.fc.in_features)

model.act3 = torch.nn.ReLU()



model.dd2 = torch.nn.Dropout(p=0.5)



model.fc3 = torch.nn.Linear(model.fc.in_features, 100)

model.act4 = torch.nn.ReLU()



model.fc4 = torch.nn.Linear(100, 2)







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0001)

# optimizer = torch.optim.RMSprop(model.parameters(), lr=1.0e-3)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)









# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
train_model(model, loss, optimizer, scheduler, num_epochs=100);
plt.plot(test_accuracy_history, label='test_acc')

plt.plot(val_accuracy_history, label='val_acc')

plt.legend()

plt.title('Accuracy');
plt.plot(test_loss_history, label='test_loss')

plt.plot(val_loss_history, label='val_loss')

plt.legend()

plt.title('Loss');
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