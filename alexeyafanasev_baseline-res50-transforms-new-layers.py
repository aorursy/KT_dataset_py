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
class BackgroundRemover:

       

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

    

def remove_background(image_roots):



    remove_photo_background = BackgroundRemover()



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

   



def unzip_data(zip_file, destination_dir):

    """Extract pictures from zip file.

    """

    print('Data extraction started...', end='')

    with zipfile.ZipFile(zip_file, 'r') as zip_obj:

        zip_obj.extractall(destination_dir)

    print('done.')

    print(f'Files unzipped to \'{destination_dir}\'\n')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import zipfile

with zipfile.ZipFile('../input/plates.zip', 'r') as zip_obj:

   # Extract all the contents of zip file in current directory

   zip_obj.extractall('/kaggle/working/')

    

print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
data_root = '/kaggle/working/plates/'

print(os.listdir(data_root))
import shutil 

import cv2



from tqdm import tqdm





class_names = ['cleaned', 'dirty']

train_dir = 'train'

valid_dir = 'valid'

test_dir = 'test'

 

data_root = '/kaggle/working/plates/'

unzip_data(zip_file='../input/plates.zip', destination_dir='/kaggle/working/')





remove_background(image_roots=[os.path.join(data_root, train_dir, 'cleaned/'),

                               os.path.join(data_root, train_dir, 'dirty/'),

                               os.path.join(data_root, 'test/')])



# Create extra images for training models

make_extra_images(image_roots=[os.path.join(data_root, train_dir, 'cleaned/'),

                               os.path.join(data_root, train_dir, 'dirty/')])





for dir_name in [train_dir, valid_dir]:

    for class_name in class_names:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):

        if i % 6 != 0:

            dest_dir = os.path.join(train_dir, class_name) 

        else:

            dest_dir = os.path.join(valid_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
!ls train
import torch

import numpy as np

import torchvision

import matplotlib.pyplot as plt

import time

import copy

from statistics import mean



from torchvision import transforms, models

train_transforms = transforms.Compose([

    transforms.CenterCrop(224),

    transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),

    transforms.RandomVerticalFlip(),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(50),

    transforms.ToTensor(),

#    lambda x: x[np.random.permutation(3), :, :],

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



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





    

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

val_dataset = torchvision.datasets.ImageFolder(valid_dir, val_transforms)



batch_size = 4

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
def train_model(model, loss, optimizer, scheduler, num_epochs):

    train_acc = []

    val_acc = []

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

            if phase == 'train':

                train_acc.append(epoch_acc.item())

            else:

                val_acc.append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)



    return model, train_acc, val_acc
model = models.resnet50(pretrained=True)



# Disable grad for all conv layers

for param in model.parameters():

    param.requires_grad = False



model.fc = torch.nn.Linear(model.fc.in_features, 100)

model.act = torch.nn.ReLU()

model.fc1 = torch.nn.Linear(100, 100)

model.act1 = torch.nn.ReLU()

model.fc2 = torch.nn.Linear(100,100)

model.act2 = torch.nn.Sigmoid()

model.fc3 = torch.nn.Linear(100,4)

model.act3 = torch.nn.Sigmoid()

model.fc4 = torch.nn.Linear(model.fc.in_features, 2)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
import random

seed = 42

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True





model, train_acc, val_acc = train_model(model, loss, optimizer, scheduler, num_epochs=100);

print(np.mean(train_acc))
tr, = plt.plot(train_acc, label='Train')

#tr_avg_acc = mean(train_acc)

val, = plt.plot(val_acc, label = 'Val')

plt.legend(handles=[tr, val])

plt.show()
test_dir = 'test'

shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

    

test_dataset = ImageFolderWithPaths('/kaggle/working/test', transform = None)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=1, shuffle=False, num_workers=0)
test_dataset
model.eval()



test_predictions = []

test_img_paths = []



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

    

test_predictions = np.concatenate(test_predictions)
inputs, labels, paths = next(iter(test_dataloader))



for img, pred in zip(inputs, test_predictions):

    show_input(img, title=pred)
submission_df = pd.DataFrame(data)
submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')

submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

submission_df.set_index('id', inplace=True)



submission_df.drop(df.columns[:-1], axis='columns', inplace=True)



submission_df.head()
submission_df.to_csv('submission.csv')
!rm -rf train val test