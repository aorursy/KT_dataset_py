import numpy as np

import pandas as pd

import torch

import torchvision

import matplotlib.pyplot as plt

import os

import cv2

import time

import copy

import shutil 

import zipfile

from torchvision import transforms, models

from tqdm import tqdm
class MyRemoveBackground:

    """Remove images background.

    """

    

    def __init__(self):

        pass



    def __call__(self, in_img):

        

        # Convert PIL image to numpy array

        in_img = np.array(in_img)

        

        # Get the height and width from OpenCV image

        height, width = in_img.shape[:2]

        

        # Create a mask holder

        mask = np.zeros([height, width], np.uint8)



        # Grab Cut the object

        bgdModel = np.zeros((1, 65),np.float64)

        fgdModel = np.zeros((1, 65),np.float64)



        # Hard Coding the Rect The object must lie within this rect.

        rect = (15, 15, width-30, height-30)

        cv2.grabCut(in_img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

        mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        out_img = in_img * mask[:, :, np.newaxis]



        # Get the background

        background = in_img - out_img



        # Change all pixels in the background that are not black to white

        background[np.where((background > [0, 0, 0]).all(axis = 2))] = [255, 255, 255]



        #Add the background and the image

        out_img = background + out_img



        return transforms.functional.to_pil_image(out_img)

    



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path
def unzip_data(zip_file, destination_dir):

    """Extract pictures from zip file.

    """

    print('Data extraction started...', end='')

    with zipfile.ZipFile(zip_file, 'r') as zip_obj:

        zip_obj.extractall(destination_dir)

    print('done.')

    print(f'Files unzipped to \'{destination_dir}\'\n')



def remove_background(image_roots):

    """Remove picture background.

       This function use MyRemoveBackground class.

    """

    remove_photo_background = MyRemoveBackground()



    print('Backgrounds removing started...')

    for path in image_roots:

        files = os.listdir(path)

        files = list(filter(lambda x: x.endswith('.jpg'), files))

        

        print(f'{len(files)} pictures was found in {path}', end='')

        for i, file in enumerate(files):

            img_original = cv2.imread(path + file)

            img_cleaned = remove_photo_background(img_original)

            img_cleaned = np.array(img_cleaned)

            cv2.imwrite(path + file, img_cleaned)

            if i % 20 == 0:

                print('\n{:>3d}/{:>3d}'.format(i, len(files)), end='')

            print('.', end='')

        print()

    print('Backgrounds removing is complete.\n')

    

def make_extra_images(image_roots):

    """Function will make extra pictures with horizontal and vertical reflection.

    """



    print('Extra pictures generation started...', end='')

    prefix_names = ['_090', '_180', '_270']



    for path in image_roots:

        files = os.listdir(path)

        files = list(filter(lambda x: x.endswith('.jpg') and '_' not in x, files))



        for i, file in enumerate(files):

            img = cv2.imread(path + file)

            # Make extra pictures: flip each of originals photo to 90, 180 and 270 degrees

            for i, angle in enumerate([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):

                img = cv2.rotate(img, angle)

                img_name = path + file[:file.find('.')] + prefix_names[i] + file[file.find('.'):]

                if not os.path.exists(img_name):

                    cv2.imwrite(img_name, img)

    print('done.')



    for path in image_roots:

        files = os.listdir(path)

        files = list(filter(lambda x: x.endswith('.jpg'), files))

        print(f'{len(files)} pictures added to \'{path}\'')

    print()

    



def make_train_valid_data():

    """Split train pictures to train and valid groups.

    """

    

    print('Copy pictures to Train and Valid groups started...', end='')

    for dir_name in [train_dir, valid_dir]:

        for class_name in class_names:

            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



    for class_name in class_names:

        src_dir = os.path.join(data_root, 'train', class_name)

        files = os.listdir(src_dir)

        files = list(filter(lambda x: x.endswith('.jpg'), files))

        

        for i, file_name in enumerate(files):

            if i % 6 != 0:

                dst_dir = os.path.join(train_dir, class_name) 

            else:

                dst_dir = os.path.join(valid_dir, class_name)

            shutil.copy(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))

    print('done.')



    # Generate mini report

    for dir_name in [train_dir, valid_dir]:

        for class_name in class_names:

            dst_dir = os.path.join(dir_name, class_name)

            files = os.listdir(dst_dir)

            files = list(filter(lambda x: x.endswith('.jpg'), files))

            print(f'{len(files)} pictures copied to \'{dst_dir}\'')

    print()

    



def make_test_data():

    """Copy test pictures to test group.

    """  



    print('Copy pictures to Test group started...', end='')

    src_dir = os.path.join(data_root, 'test')

    dst_dir = os.path.join(test_dir, 'unknown')

    shutil.copytree(src_dir, dst_dir)

    files = os.listdir(dst_dir)

    files = list(filter(lambda x: x.endswith('.jpg'), files))

    print('done.')

    print(f'{len(files)} pictures copied to \'{dst_dir}\'\n')
# Clear output directory

!rm * --recursive         



# Let's define some variables

class_names = ['cleaned', 'dirty']

train_dir = 'train'

valid_dir = 'valid'

test_dir = 'test'



# Extract images (Kaggle enviropment)

data_root = '/kaggle/working/plates/'

unzip_data(zip_file='../input/plates.zip', destination_dir='/kaggle/working/')



# Remove images background 

remove_background(image_roots=[os.path.join(data_root, train_dir, 'cleaned/'),

                               os.path.join(data_root, train_dir, 'dirty/'),

                               os.path.join(data_root, 'test/')])



# Create extra images for training models

make_extra_images(image_roots=[os.path.join(data_root, train_dir, 'cleaned/'),

                               os.path.join(data_root, train_dir, 'dirty/')])



# Copy images to train, valid and test dir

make_train_valid_data()

make_test_data()

# Transformations methods

image_transforms = { 

    'train': transforms.Compose([

        transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),

        transforms.RandomChoice([transforms.CenterCrop(180),

                                 transforms.CenterCrop(160),

                                 transforms.CenterCrop(140),

                                 transforms.CenterCrop(120),

                                 transforms.Compose([transforms.CenterCrop(280),

                                                     transforms.Grayscale(3),

                                                     ]),

                                 transforms.Compose([transforms.CenterCrop(200),

                                                     transforms.Grayscale(3),

                                                     ]),

                                 ]),

        transforms.Resize((224, 224)),

        transforms.ColorJitter(hue=(0.1, 0.2)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),

        transforms.RandomChoice([transforms.CenterCrop(180),

                                 transforms.CenterCrop(160),

                                 transforms.CenterCrop(140),

                                 transforms.CenterCrop(120),

                                 transforms.Compose([transforms.CenterCrop(280),

                                                     transforms.Grayscale(3),

                                                     ]),

                                 transforms.Compose([transforms.CenterCrop(200),

                                                     transforms.Grayscale(3),

                                                     ]),

                                 ]),

        transforms.Resize((224, 224)),

        transforms.ColorJitter(hue=(0.1, 0.2)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ]),}



# Datasets

dataset = {

    'train': torchvision.datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),

    'valid': torchvision.datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),

    'test': ImageFolderWithPaths('/kaggle/working/test', transform=None),

}

 

# Dataloaders

batch_size = 4

train_dataloader = torch.utils.data.DataLoader(dataset['train'],

                                               batch_size=batch_size,

                                               shuffle=True,

                                               num_workers=batch_size)



valid_dataloader = torch.utils.data.DataLoader(dataset['valid'],

                                               batch_size=batch_size,

                                               shuffle=True,

                                               num_workers=batch_size)



test_dataloader  = torch.utils.data.DataLoader(dataset['test'],

                                               batch_size=1,

                                               shuffle=False,

                                               num_workers=0)



# Mini report of loaded data

print('Train, Valid and Test datasets are loaded:\n')

print('{:<7s}{:>10s}{:>10s}'.format('Dataset', 'Batches', 'Pictures')), print('-' * 28)

print('{:<7s}{:>10d}{:>10d}'.format('Train', len(train_dataloader), len(dataset['train'])))

print('{:<7s}{:>10d}{:>10d}'.format('Valid', len(valid_dataloader), len(dataset['valid'])))

print('{:<7s}{:>10d}{:>10d}'.format('Test',  len(test_dataloader),  len(dataset['test'])))

                            
%matplotlib inline

plt.rcParams['figure.figsize'] = (14, 10)



image_iter = iter(train_dataloader)

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])



for i in range(3):

    X_batch, y_batch = next(image_iter)

    for j in range(0, len(X_batch)):

        plt.subplot(3, 4, i*batch_size + j + 1)

        plt.title(class_names[y_batch[j].item()])

        plt.imshow(X_batch[j].permute(1, 2, 0).numpy() * std + mean)

  
class MyResNet18(torch.nn.Module):

    def __init__(self):

        super(MyResNet18, self).__init__()

        self.net = models.resnet18(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.fc.in_features

        self.net.fc = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 256),

            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),

            torch.nn.Sigmoid(),

            torch.nn.Linear(128, 2),

            # nn.LogSoftmax(dim=1) # For using NLLLoss()

        )  

    def forward(self, x):

        x = self.net(x)

        return x



    

class MyResNet50(torch.nn.Module):

    def __init__(self):

        super(MyResNet50, self).__init__()

        self.net = models.resnet50(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.fc.in_features

        self.net.fc = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 256),

            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),

            torch.nn.Sigmoid(),

            # torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 2),

            # torch.nn.LogSoftmax(dim=1) # For using NLLLoss()

        )  

    def forward(self, x):

        x = self.net(x)

        return x



    

class MyResNet152(torch.nn.Module):

    def __init__(self):

        super(MyResNet152, self).__init__()

        self.net = models.resnet152(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.fc.in_features

        self.net.fc = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 256),

            torch.nn.ReLU(),

            torch.nn.Linear(256, 2),

            # nn.LogSoftmax(dim=1) # For using NLLLoss()

        )  

    def forward(self, x):

        x = self.net(x)

        return x

    



class MyMobilenet(torch.nn.Module):

    def __init__(self):

        super(MyMobilenet, self).__init__()

        self.net = models.mobilenet_v2(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.classifier[1].in_features

        self.net.classifier[1] = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 256),

            torch.nn.ReLU(),

            # torch.nn.Dropout(0.2),

            torch.nn.Linear(256, 2),

            # torch.nn.LogSoftmax(dim=1) # For using NLLLoss()

        )  

    def forward(self, x):

        x = self.net(x)

        return x    





class MyAlexNet(torch.nn.Module):

    def __init__(self):

        super(MyAlexNet, self).__init__()

        self.net = models.alexnet(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.classifier[6].in_features

        self.net.classifier[6] = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 2),

            # torch.nn.ReLU(),

            # torch.nn.Linear(256, 2),

            # torch.nn.LogSoftmax(dim=1) # For using NLLLoss()

        )  

    def forward(self, x):

        x = self.net(x)

        return x  
def train_model(model, loss, optimizer, scheduler, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)



    loss_hist = {'train':[], 'valid':[]}

    accuracy_hist = {'train':[], 'valid':[]}

    

    print('{:<7s}|{:^20s}|{:^20s}|'.format('', 'Train', 'Valid'))

    print('{:<7s}|{:>10s}{:>10s}|{:>10s}{:>10s}|'.format('Epoch', 'Loss', 'Acc', 'Loss', 'Acc'))

    print('-' * 50)

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                dataloader = train_dataloader

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                dataloader = valid_dataloader

                model.eval()   # Set model to evaluate mode



            running_loss = 0.

            running_acc = 0.



            # Iterate over data.

            for inputs, labels in dataloader:

                inputs = inputs.to(device)

                labels = labels.to(device)



                optimizer.zero_grad()



                # Forward and backward

                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)

                    loss_value = loss(preds, labels)

                    preds_class = preds.argmax(dim=1)



                    # Backward + optimize only if in training phase

                    if phase == 'train':

                        loss_value.backward()

                        optimizer.step()



                # Statistics

                running_loss += loss_value.item()

                # Running_acc += (preds_class == labels.data).float().mean()

                running_acc += (preds_class == labels.data).float().mean().data.cpu().numpy()                





            epoch_loss = running_loss / len(dataloader)

            epoch_acc = running_acc / len(dataloader)

            if phase == 'train':

                print('{:>3d}/{:>3d}|{:>10.4f}{:>10.4f}|'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc), end='')

            else:

                print('{:>10.4f}{:>10.4f}|'.format(epoch_loss, epoch_acc))



            loss_hist[phase].append(epoch_loss)

            accuracy_hist[phase].append(epoch_acc)



    return model, loss_hist, accuracy_hist
def train_MyModel(model, epoch_num):

    print('\n' + model.__class__.__name__ + ' training with {} epochs started...\n'.format(epoch_num))

 

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=1.0e-3)



    # Decay LR by a factor of 0.1 every 7 epochs

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    model, losses, accuracies = train_model(model, loss, optimizer, scheduler, num_epochs=epoch_num);

    print('\nModel training finished.')    

    

    return model, losses, accuracies

import random

seed = 77

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True



# model = MyResNet18()

model = MyResNet50()

# model = MyResNet152()

# model = MyMobilenet()

# model = MyAlexNet()



# Run model training

model, losses, accuracies = train_MyModel(model, epoch_num=30)



# Switch model to prediction mode

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model accuracies visualization

plt.rcParams['figure.figsize'] = (14, 7)

for experiment_id in accuracies.keys():

    plt.plot(accuracies[experiment_id], label=experiment_id)

plt.legend(loc='upper left')

plt.title('Model Accuracy')

plt.xlabel('Epoch num', fontsize=15)

plt.ylabel('Accuracy value', fontsize=15);

plt.grid(linestyle='--', linewidth=0.5, color='.7')
# Model loss-function visualization

plt.rcParams['figure.figsize'] = (14, 7)

for experiment_id in losses.keys():

    plt.plot(losses[experiment_id], label=experiment_id)

plt.legend(loc='upper left')

plt.title('Model Loss')

plt.xlabel('Epoch num', fontsize=15)

plt.ylabel('Loss function value', fontsize=15)

plt.grid(linestyle='--', linewidth=0.5, color='.7')
transform_image = {

    'to_tensor_and_normalize': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ])

}



# List of transformation methods

transforms_list = { 

    'original': transforms.Compose([

        transforms.Resize((224, 224)),

    ]),   

#     'crop_220': transforms.Compose([

#         transforms.CenterCrop(220),

#         transforms.Resize((224, 224)),

#     ]), 

#     'crop_200': transforms.Compose([

#         transforms.CenterCrop(200),

#         transforms.Resize((224, 224)),

#     ]),    

    'crop_180': transforms.Compose([

        transforms.CenterCrop(180),

        transforms.Resize((224, 224)),

    ]),    

    'crop_160': transforms.Compose([

        transforms.CenterCrop(160),

        transforms.Resize((224, 224)),

    ]),   

    'crop_140': transforms.Compose([

        transforms.CenterCrop(140),

        transforms.Resize((224, 224)),

    ]),   

#     'crop_120': transforms.Compose([

#         transforms.CenterCrop(120),

#         transforms.Resize((224, 224)),

#     ]),    

    'gray_280': transforms.Compose([

        transforms.Grayscale(3),

        transforms.CenterCrop(280),

        transforms.Resize((224, 224)),

    ]),

    'gray_200': transforms.Compose([

        transforms.Grayscale(3),

        transforms.CenterCrop(200),

        transforms.Resize((224, 224)),

    ]),

    'r_crop_180_1': transforms.Compose([

        transforms.RandomCrop(180),

        transforms.Resize((224, 224)),

    ]),

    'r_crop_180_2': transforms.Compose([

        transforms.RandomCrop(180),

        transforms.Resize((224, 224)),

    ]),

    'r_crop_180_3': transforms.Compose([

        transforms.Grayscale(3),

        transforms.RandomCrop(180),

        transforms.Resize((224, 224)),

    ]),        

}



# Test Dataset

dataset['test'] = ImageFolderWithPaths('/kaggle/working/test', transform=None)



# Test Dataloaders

test_dataloader  = torch.utils.data.DataLoader(dataset['test'],

                                               batch_size=1,

                                               shuffle=False,

                                               num_workers=0)
img_id = 50



img_original = test_dataloader.dataset[img_id][0]



img_id = test_dataloader.dataset[img_id][2]

img_id = img_id.replace('/kaggle/working/test/unknown/', '')

img_id = img_id.replace('.jpg', '')



labels = {}

labels['id'] = img_id



# Make prediction for each method in tranformation_list

for i, method in enumerate(transforms_list):

    img_transformed = transforms_list[method](img_original)

    tensor = transform_image['to_tensor_and_normalize'](img_transformed)

    tensor = tensor.to(device)

    tensor = tensor.unsqueeze(0)



    with torch.set_grad_enabled(False):

        preds = model(tensor)

        

    label = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()[0]

    labels[method] = label



# Vizualization 

plt.rcParams['figure.figsize'] = (15, 10)

for i, method in enumerate(transforms_list):

    img_transformed = transforms_list[method](img_original)

    plt.subplot(3, 5, i + 1)

    plt.title(img_id +': ['+ method + ', ' + str(round(labels[method],4))+']')

    plt.imshow(img_transformed); 



# Print prediction's results    

print('{:3s}{:15s}{:7s}'.format('N', 'Method', 'Percent')), print('-' * 25)

for i, method in enumerate([x for x in labels if x != 'id']):

    print('{:<3d}{:<15s}{:>7.4f}'.format(i+1, method, labels[method]))

print('-' * 25)
random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True



data = []

for img_original, labels, img_id in tqdm(test_dataloader.dataset):

    labels = {}

    labels['id'] = img_id

    probs = np.array([])



    for i, method in enumerate(transforms_list):

        img_transformed = transforms_list[method](img_original)

        tensor = transform_image['to_tensor_and_normalize'](img_transformed)

        tensor = tensor.to(device)

        tensor = tensor.unsqueeze(0)



        with torch.set_grad_enabled(False):

            preds = model(tensor)



        label = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()[0]

        labels[method] = label



    data.append(labels)
# Create DataFrame

df = pd.DataFrame(data)



df['id'] = df['id'].str.replace('/kaggle/working/test/unknown/', '')

df['id'] = df['id'].str.replace('.jpg', '')



df.set_index('id', inplace=True)

df.head()
# Probability distribution visualization

plt.rcParams['figure.figsize'] = (15, 10)

for i, col_name in enumerate(df.columns):

    plt.subplot(3, 5, i + 1)

    plt.title(col_name)

    plt.hist(x=[df[col_name]], bins=100, histtype='bar');

# Data is skewed.So let's make some modifications to fix it

df['original'] = df['original']**(0.6)

# df['crop_220'] = df['crop_220']**(2.0)

# df['crop_200'] = df['crop_200']**(2.0)

df['crop_180'] = df['crop_180']**(2.5)

df['crop_160'] = df['crop_160']**(2.5)

df['crop_140'] = df['crop_140']**(2.5)

# df['crop_120'] = df['crop_120']**(2.0)

df['gray_280'] = df['gray_280']**(2.0)

df['gray_200'] = df['gray_200']**(2.5)

df['r_crop_180_1'] = df['r_crop_180_1']**(2.0)

df['r_crop_180_2'] = df['r_crop_180_2']**(2.0)

df['r_crop_180_3'] = df['r_crop_180_3']**(2.0)
# And a middle of each prediction method to zero

for col_name in df.columns:

    gap = 0.05

    plates_min = 999



    # Serch a middle between 0.4 and 0.7

    for i in range(40, 70):

        plates_num = df[(df[col_name] > i/100) & (df[col_name] < i/100 + gap)][col_name].count()

        if plates_min > plates_num:

            plates_min = plates_num

            middle = i/100



    # Shift a middle to zero

    df[col_name] = df[col_name] - middle + gap/2
# Let's see what we've got

plt.rcParams['figure.figsize'] = (15, 10)

for i, col_name in enumerate(df.columns):

    plt.subplot(3, 5, i + 1)

    plt.title(col_name)

    plt.hist(x=[df[col_name]], bins=100, histtype='bar');
# Prepare submission file

df['mean'] = df.mean(axis=1)

df['label'] = df['mean'].map(lambda x: 'cleaned' if x < 0 else 'dirty')

df.head()
# Remove extra columns

df.drop(df.columns[:-1], axis='columns', inplace=True)

df.head()
# Export predictions to *.csv file

# file_name = 'submission_' + model.__class__.__name__ + '.csv'

file_name = 'submission.csv'

df.to_csv(file_name)
!rm -rf train valid test