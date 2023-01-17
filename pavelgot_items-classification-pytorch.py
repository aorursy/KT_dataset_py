#Importing modules
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from PIL import Image
import os
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
#If you run this cell in  as a Kaggle's notebook:
DATASET_PATH = "/kaggle/input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/"
print(os.listdir(DATASET_PATH))
# Reading the rows and dropping the ones with errors
df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=44416, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
df.head(5)
plt.figure(figsize=(7,20))
df.articleType.value_counts().sort_values().plot(kind='barh')
N_Pictures = 250
N_Classes = np.sum(df.articleType.value_counts().to_numpy() > N_Pictures)
#Number of classes with sufficient images to train on:
N_Classes
#Inspecting the item classes that made it to our new dataset
temp = df.articleType.value_counts().sort_values(ascending=False)[:N_Classes]
temp[-5:]
#Saving item types(labels) with their counts
items_count = temp.values
items_label = temp.index.tolist()
#Creating new dataframes for training/validation
df_train = pd.DataFrame(columns=['articleType','image'])
df_val   = pd.DataFrame(columns=['articleType','image'])


for ii in range(0,N_Classes):
    
    #print(items_label[ii])
    
    temp = df[df.articleType==items_label[ii]].sample(N_Pictures)

    df_train = pd.concat([df_train, temp[ :int(N_Pictures*0.6) ][['articleType','image']] ]            , sort=False)
    df_val   = pd.concat([df_val,   temp[  int(N_Pictures*0.6): N_Pictures ][['articleType','image']] ], sort=False)

df_train.reset_index(drop=True)
df_val.reset_index(drop=True)
#Create folders for new dataset
os.chdir(r'/kaggle/working/')
os.mkdir('data')
os.mkdir('data/train')
os.mkdir('data/val')
os.chdir(r'/kaggle/working/')

data = {'train': df_train, 'val': df_val}

# and save each individual image to the new directory
for x in ['train','val']:
    
    print(x)
    
    for label, file in data[x].values:
        
        try:
            img = Image.open(DATASET_PATH+'images/'+file)
        except FileNotFoundError:
            # If file does not exist continue
            continue
            
        #Else save file to new directory  
        try:
            img.save('data/'+x+'/'+label+'/'+file) 

        except FileNotFoundError:
            #If folder does not exist, create one and save the image
            os.mkdir('data/'+x+'/'+label)
            img.save('data/'+x+'/'+label+'/'+file)
            print(label,end=' ')
    
#Inspect if all the folders have been created
%ls data/train/
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

#Changing the number of outputs in the last layer to the number of different item types
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)
#Saving our model's weights: 

%mkdir model
torch.save(model_ft.state_dict(), 'model/model_fine_tuned.pt')
%ls 

#Download the model weights and save them locally
from IPython.display import FileLink
FileLink(r'model/model_fine_tuned.pt')
visualize_model(model_ft)
model_conv = torchvision.models.resnet18(pretrained=True)
# Freezing the weights
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=15)
#Saving our model's weights: 

%mkdir model
torch.save(model_conv.state_dict(), 'model/model_fixed_feature.pt')
%ls model/

from IPython.display import FileLink
FileLink(r'model/model_fixed_feature.pt')
visualize_model(model_conv)

plt.ioff()
plt.show()
import shutil
import requests

%mkdir test_data

#Download the images from internet:
urls = [r'https://media.karousell.com/media/photos/products/2017/02/19/bnib_ecco_formal_shoes_brown_1487494013_3659d7da.jpg',
        r'https://17pprhpagc13i5210btdmqmf-wpengine.netdna-ssl.com/wp-content/uploads/2020/06/wallets-vegancom.jpg',
        r'https://www.tudorwatch.com/-/media/model-assets/wrist/l/tudor-m79030n-0001.jpg',
        r'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/gettyimages-1256162242.jpg?crop=0.889xw:1.00xh;0,0&resize=640:*',
        r'https://cdn.shopify.com/s/files/1/0285/2916/4420/products/P8501_20PORTER_20JEAN_20MODEL_20_D_1024x1024.jpg?v=1584339217',
        r'https://images1.novica.net/pictures/26/p350363_2.jpg',
        r'https://cdn.shopify.com/s/files/1/0021/6331/0691/products/DP0423HS_43ddebc5-5e53-4dd6-af00-76ff2970bbd2_grande.jpg?v=1586232410',
        r'https://goodhousekeeping.fetcha.co.za/wp-content/uploads/2019/03/sweat.jpg',
        r'https://sneakernews.com/wp-content/uploads/2020/08/undefeated-kobe-5-protro-gold-purple-DA6809-700-3.jpg']

for ii, url in enumerate(urls):
    response = requests.get(url, stream=True)
    with open('test_data/img_test_' + str(ii) + '.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    
%ls test_data
from PIL import Image

model = model_ft


model.eval()
fig = plt.figure(figsize=(10,18))

with torch.no_grad():

    for ii, file_name in enumerate(os.listdir( 'test_data' )):
        img = Image.open( 'test_data' + '/' + file_name)
        img_t = data_transforms['val'](img).unsqueeze(0)
        img_t = img_t.to(device)

        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        
        ax = plt.subplot(len(os.listdir( 'test_data' )),3, ii+1)
        ax.axis('off')
        ax.set_title('predicted: {}'.format(class_names[int(preds.cpu().numpy())]))
        plt.imshow( np.array(img) )
        