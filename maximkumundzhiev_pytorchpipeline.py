# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import shutil

import os



class BaseLine:

    def __init__(self):

        self.input_images_path = '/kaggle/input/physdl-201920-second-challenge/images/images'

        self.csv = '/kaggle/input/physdl-201920-second-challenge/train.csv'

        

        self.input_path = '/kaggle/input/physdl-201920-second-challenge'

        self.train_path = '/kaggle/train'

        self.test_path = '/kaggle/test'

        self.valid_path = '/kaggle/validation'



    

    def folder_generator(self):

        import pandas as pd

        import logging as log

        import os

        

        csv_labels = pd.read_csv(self.csv)

        

        images_paths = list(csv_labels['file'])

        images_paths = [path.replace('images/', '') for path in images_paths]

        

        images_labels = list(csv_labels['is_tiger'])

        counter = 1

        

        tiger_path_train = os.path.join(self.train_path, 'tiger')

        not_tiger_path_train = os.path.join(self.train_path, 'not_tiger')

        

        tiger_path_val = os.path.join(self.valid_path, 'tiger')

        not_tiger_path_val = os.path.join(self.valid_path, 'not_tiger')

        

        test_unknown_path = os.path.join(self.test_path, 'unknown')

        

        os.mkdir(self.train_path)

        os.mkdir(tiger_path_train)

        os.mkdir(not_tiger_path_train)



        os.mkdir(self.valid_path)

        os.mkdir(tiger_path_val)

        os.mkdir(not_tiger_path_val)

        

        os.mkdir(self.test_path)

        os.mkdir(test_unknown_path)

        

        #Handling Train Images

        print('###Handling Train Images')

        for image_name, image_label in zip(images_paths, images_labels):

            print('Coping image: {}, with label: {}'.format(image_name, image_label))    

            if image_name.split('_')[0] == 'train' and counter % 5 != 0:

                if image_label == 1:

                    source_dir = os.path.join(self.input_images_path, image_name) 

                    dest_dir = os.path.join(tiger_path_train, image_name)

                    shutil.copy(source_dir, dest_dir)

                    counter += 1

                    print('Coped image: {} to {}'.format(image_name, tiger_path_train))

                else:

                    source_dir = os.path.join(self.input_images_path, image_name) 

                    dest_dir = os.path.join(not_tiger_path_train, image_name)

                    shutil.copy(source_dir, dest_dir)

                    counter += 1

                    print('Coped image: {} to {}'.format(image_name, not_tiger_path_train))

            else:

                if image_label == 1:

                    source_dir = os.path.join(self.input_images_path, image_name) 

                    dest_dir = os.path.join(tiger_path_val, image_name)

                    shutil.copy(source_dir, dest_dir)

                    counter += 1

                    print('Coped image: {} to {}'.format(image_name, tiger_path_val))

                else:

                    source_dir = os.path.join(self.input_images_path, image_name) 

                    dest_dir = os.path.join(not_tiger_path_val, image_name)

                    shutil.copy(source_dir, dest_dir)

                    counter += 1

                    print('Coped image: {} to {}'.format(image_name, not_tiger_path_val))

            

        #Handling Test Images

        print('###Handling Test Images')

        list_of_images = os.listdir(self.input_images_path)

        for image_name in list_of_images:

            if image_name.split('_')[0] == 'test':

                source_dir = os.path.join(self.input_images_path, image_name) 

                dest_dir = os.path.join(test_unknown_path, image_name)

                shutil.copy(source_dir, dest_dir)

                print('Coped image: {} to {}'.format(image_name, test_unknown_path))

                      

        print('Number of All train images: {}'.format(len(os.listdir(tiger_path_train)) + len(os.listdir(not_tiger_path_train))))

        print('Number of All validation images: {}'.format(len(os.listdir(tiger_path_val)) + len(os.listdir(not_tiger_path_val))))

        

        print('Number of All test images: {}'.format(len(os.listdir(self.test_path))))

        

        print('Tiger Labels: {}'.format(len([label for label in images_labels if label == 1])))

        print('NotTiger Labels: {}'.format(len([label for label in images_labels if label != 1])))

    

    def data_loader(self):

        import torch

        import torchvision

        from torchvision import transforms

        #ImageFolder

        #Augementation for train and valid images

        train_transforms = transforms.Compose([

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])



        val_transforms = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(), 

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])



        train_dataset = torchvision.datasets.ImageFolder(self.train_path, train_transforms)

        validation_dataset = torchvision.datasets.ImageFolder(self.valid_path, val_transforms)

        

        #DataLoader

        batch_size = 16

        train_dataloader = torch.utils.data.DataLoader(

            train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size

        )



        validation_dataloader = torch.utils.data.DataLoader(

            validation_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size

        )

        print(len(train_dataset), len(validation_dataset))

        print(len(train_dataloader), len(validation_dataloader))

        return train_dataloader, validation_dataloader

    

    def show_images(self, train_dataloader):

        import numpy as np

        import matplotlib.pyplot as plt

        X_batch, y_batch = next(iter(train_dataloader))

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([[0.229, 0.224, 0.225]])

        for index in range(5):

            plt.title(y_batch[index])

            plt.imshow(X_batch[index].permute(1,2,0).numpy() * std + mean, )

            plt.show()

    

    def model(self):

        from torchvision import models

        import torch

        model = models.resnet50(pretrained=True)

        

        #Disable grad for all conv layers

        for param in model.parameters():

            param.requires_grard = False 

    

        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        

        loss = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(),amsgrad=True, lr=1.0e-3)

        

        #Declay LR by a factor of 0.1 every 7th epoch

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

        return model, loss, optimizer, scheduler, device

    

    def train_model(self, model, loss, optimizer, scheduler, train, validation, device, num_epochs):

        import torch

        from tqdm import tqdm

        for epoch in range(num_epochs):

            print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)



            # Each epoch has a training and validation phase

            for phase in ['train', 'val']:

                if phase == 'train':

                    dataloader = train

                    scheduler.step()

                    model.train()  # Set model to training mode

                else:

                    dataloader = validation

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



                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)



        return model



    

    def run(self):

        #Processing data

        Job = BaseLine()

        Job.folder_generator()

        train_dataloader, validation_dataloader = Job.data_loader()

        Job.show_images(train_dataloader)

        

        #Define model

        model, loss, optimizer, scheduler, device = Job.model()

        #print(model, loss, optimizer, scheduler)

        #Train model 

        trained_model = Job.train_model(model, loss, optimizer, scheduler, train_dataloader, validation_dataloader, device, num_epochs=5)

        return trained_model
model = BaseLine().run()
import numpy as np

import torch

import torchvision



from torchvision import transforms

from tqdm import tqdm



class ImageFolderWithPath(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPath, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path



test_dir = '/kaggle/test'



val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(), 

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])



batch_size = 16     

test_dataset = ImageFolderWithPath(test_dir, val_transforms)



test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size

)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
import matplotlib.pyplot as plt



inputs, labels, paths = next(iter(test_dataloader))

mean = np.array([0.485, 0.456, 0.406])

std = np.array([[0.229, 0.224, 0.225]])



def show_input(input_tensor, title=''):

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()

    plt.pause(0.001)



for img, pred in zip(inputs, test_predictions):

    show_input(img, title=pred)
import pandas as pd

files = [path.replace('/kaggle/test/unknown/', 'images/') for path in test_img_paths]

submission_df = pd.DataFrame.from_dict({'file': files, 'is_tiger': test_predictions})

submission_df.to_csv('submission.csv', index=False)