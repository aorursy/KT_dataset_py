# Import, configuration cell



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # simple matplotlib interface for data visualization



import os # to work with directories

import zipfile # to extract images from input .zip file

import shutil  # to copy data between folders



from tqdm import tqdm # loops progress bar 



import torch # main machine learning framework

import imgaug.augmenters as iaa # image augmentation library

import torchvision # image loader, augmentation, neural networks architectures

from torchvision import transforms, models





# Configuration for the kernel

config = {

    "n_train_epoch": 65,

    "batch_size": 8,

    "scheduler_use": False,

    "n_augmentation_repeat": 8,

    "show_train_progress": True,

    "show_augmented_images": True

}
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.



print('Contents of the input folder: \n', 

      os.listdir("../input/"))



# Extract all the contents of zip file in current directory

with zipfile.ZipFile('../input/plates.zip', 'r') as zip_obj:

    zip_obj.extractall('/kaggle/working/')

    

print('After zip extraction: \n', 

      os.listdir("/kaggle/working/"))
data_root = '/kaggle/working/plates/'

train_dir = 'train'

val_dir = 'val'



class_names = ['cleaned', 'dirty']



# Creating 'cleaned' and 'dirty' folders inside the 'train' and 'val' directories

for dir_name in [train_dir, val_dir]:

    for class_name in class_names:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



# Copy images from the 'train' folder to the new 'train' folder considering the labels

# Each sixth image goes to validation data

for class_name in class_names:

    source_dir = os.path.join(data_root, 'train', class_name)

    for i, file_name in enumerate(tqdm(sorted(os.listdir(source_dir)))):

        dest_dir = os.path.join(train_dir if i%5!=0 else val_dir, class_name)

        shutil.copy(os.path.join(source_dir, file_name), 

                        os.path.join(dest_dir, file_name))
!ls train/cleaned
# Train augmentation cell with imgaug library.

# Unfortunaly, this library doesn't updated on kaggle, so i couldn't use a lot of important functions.

# This cell isn't used, but just i just left it here.



# random_using(0.45, ...) applies the given augmenter in 45% of all cases.

random_using = lambda aug: iaa.Sometimes(0.45, aug)



class ImgAugTransform:

    def __init__(self):

        self.aug = iaa.Sequential(

            [

                # Apply the following augmenters to most images

                random_using(iaa.CropAndPad(percent=(-0.1, 0.3))),

                iaa.Fliplr(0.5), # horizontally flip 50% of all images

                iaa.Flipud(0.2),  # vertically flip 20% of all images

                random_aug_use(iaa.Affine(

                    rotate=(-90, 90), # rotate by -90 to +45 degrees

                )),

                iaa.Add((30, 30)), # change brightness of images (by -30 to 30 of original value) 

                iaa.AddToHueAndSaturation((-15, 15))               

            ], random_order=True)

    

    def __call__(self, img):

        img = np.array(img)

        return np.ascontiguousarray(self.aug.augment_image(img))
train_transforms = transforms.Compose([

    #ImgAugTransform(),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.35),

    transforms.RandomAffine(degrees=(-60, 60)),

    transforms.ColorJitter(brightness=(0.7, 1.35), 

                           contrast=(0.8, 1.4), 

                           saturation=(0.8, 1.2), 

                           hue=(-0.1, 0.1)),

    transforms.CenterCrop((224,224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomHorizontalFlip(p=0.2),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



train_dataset = torch.utils.data.ConcatDataset([

     torchvision.datasets.ImageFolder(train_dir, train_transforms)

    for _ in range(config['n_augmentation_repeat'])

])

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)





batch_size = config['batch_size']

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
def show_input(input_tensor, title=''):

    """

    Function for showing images. 

        

    Keyword arguments:

        input_rensor -- one torch.Tensor object

        title -- plate type (cleaned or dirty) (default '')

    """

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()

    plt.pause(0.001)
for X_batch, y_batch in tqdm(train_dataloader): 

    for x_item, y_item in zip(X_batch, y_batch):

        show_input(x_item, title=class_names[y_item])

    break
def train_model(model, loss, optimizer, scheduler, num_epochs):

    """

    Model training function. In range of epochs number it passes 2 phases: training and validation.

    If the validation phase, we don't allow the network to change.

    

    Keyword arguments:

    """

    

    global config

    statistics = {

        'val_loss': [],

        'train_loss': [],

        'val_acc': [],

        'train_acc': []

    }

    

    for epoch in range(num_epochs):

        if config['show_train_progress']:

            print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

                if config['scheduler_use']:

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

                statistics['train_loss'].append(epoch_loss)

                statistics['train_acc'].append(epoch_acc.item())

            else:

                statistics['val_loss'].append(epoch_loss)

                statistics['val_acc'].append(epoch_acc.item())

    

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)



    return (model, statistics)
model = models.mobilenet_v2(pretrained = True)



# Disable grad for all conv layers

#for param in model.parameters():

    #param.requires_grad = False



model.classifier[1] = torch.nn.Sequential(

    torch.nn.Linear(model.classifier[1].in_features, model.classifier[1].in_features),

    torch.nn.ReLU(),

    torch.nn.Dropout(p=0.5),



    torch.nn.Linear(model.classifier[1].in_features, 2)

)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
model, stats = train_model(model, 

                           loss, 

                           optimizer, 

                           scheduler, 

                           num_epochs=config['n_train_epoch'])
# Saving the model

model_filename = 'model_state_dict.pt'

torch.save(model.state_dict(), model_filename)
plt.figure(figsize=(24, 9))

function_names = ['Loss', 'Accuracy']

stats_names = iter(list(stats.keys()))

  

for i in range(2):

    ax = plt.subplot(1, 2, i+1)

    ax.plot(range(config['n_train_epoch']),

            stats[next(stats_names)], 

            label='Validation', 

            color='darkorchid', 

            lw=2.5)

    ax.plot(range(config['n_train_epoch']), 

            stats[next(stats_names)], 

            label='Training', 

            color='mediumspringgreen', 

            lw=2.5)

    ax.set_xlabel('Number of training epochs')

    ax.set_ylabel(function_names[i] + ' value')

    ax.set_title(function_names[i] + ' Functions', fontsize=20)

    ax.legend(fontsize=14)
test_dir = 'test'

shutil.copytree(os.path.join(data_root, 'test'), 

                os.path.join(test_dir, 'unknown'))
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

    

test_dataset = ImageFolderWithPaths('/kaggle/working/test', val_transforms)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataset
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
for i, (inputs, labels, paths) in enumerate(test_dataloader):

    for img, pred in zip(inputs, test_predictions):

        show_input(img, title=pred)

    if i == 1:

        break
# Creating and postprocessing data to csv file.

submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})



submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')

submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

submission_df.set_index('id', inplace=True)

submission_df.head(n=20)
submission_df.to_csv("submission.csv")
!rm -rf plates train val test