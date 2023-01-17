

import os

import shutil

from tqdm import tqdm

import torch

from torch import nn

from torch.nn import functional as F

import torchvision

from torchvision import transforms



from sklearn import metrics

from sklearn.metrics import confusion_matrix

from PIL import Image

from matplotlib import pyplot as plt



import numpy as np

import random
train_dirs = {

    'cat': '../input/dogs-cats-images/dataset/training_set/cats',

   'dog': '../input/dogs-cats-images/dataset/training_set/dogs',

}

test_dirs = {

  'cat': '../input/dogs-cats-images/dataset/test_set/cats',

  'dog': '../input/dogs-cats-images/dataset/test_set/dogs',

}
train_transform = torchvision.transforms.Compose([

    torchvision.transforms.Resize(size=(224, 224)),

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



test_transform = torchvision.transforms.Compose([

    torchvision.transforms.Resize(size=(224, 224)),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
class CatDog_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_dirs, transform):

       

        def get_images(class_name):

            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('jpg')]

            print(f'Found {len(images)} {class_name} examples')

            return images

        

        self.images = {}

        self.class_names = ['cat', 'dog']

        

        for class_name in self.class_names:

            self.images[class_name] = get_images(class_name)

            

        self.image_dirs = image_dirs

        self.transform = transform

        

    

    def __len__(self):

        return sum([len(self.images[class_name]) for class_name in self.class_names])

    

    

    def __getitem__(self, index):

        class_name = random.choice(self.class_names)

        index = index % len(self.images[class_name])

        image_name = self.images[class_name][index]

        image_path = os.path.join(self.image_dirs[class_name], image_name)

        image = Image.open(image_path).convert('RGB')

        return self.transform(image), self.class_names.index(class_name)
print("Train Dataset Info ...")

train_dataset = CatDog_Dataset(train_dirs, train_transform)

print("\n")

print("Validation Dataset Info ...")

test_dataset = CatDog_Dataset(test_dirs, test_transform)
batch_size = 8



dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers =3)

dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers =3)



print('Number of training batches', len(dl_train))

print('Number of test batches', len(dl_test))
class_names = train_dataset.class_names





def show_images(images, labels, preds):

    plt.figure(figsize=(12, 7))

    images = images.cpu()

    labels = labels.cpu()

    for i, image in enumerate(images):

        plt.subplot(1, 8, i + 1, xticks=[], yticks=[])

        image = image.numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = image * std + mean

        image = np.clip(image, 0., 1.)

        plt.imshow(image)

        col = 'green'

        if preds[i] != labels[i]:

            col = 'red'

            

        plt.xlabel(f'{class_names[int(labels[i].cpu().numpy())]}')

        plt.ylabel(f'{class_names[int(preds[i].cpu().numpy())]}', color=col)

    plt.tight_layout()

    plt.show()
images, labels = next(iter(dl_train))

show_images(images, labels, labels)
images, labels = next(iter(dl_test))

show_images(images, labels, labels)

print(labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CD_ResNet34 = torchvision.models.resnet34(pretrained=True);
#print(CD_ResNet34)
CD_ResNet34.fc = torch.nn.Linear(in_features=512, out_features=2)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(CD_ResNet34.parameters(), lr=3e-5)

CD_ResNet34 = CD_ResNet34.to(device)
print(CD_ResNet34)
# trainable parameters

for param in CD_ResNet34.parameters():

    if param.requires_grad:

        print(param.shape)
def show_preds():

    CD_ResNet34.eval()

    images, labels = next(iter(dl_test))

    images = images.to(device)

    labels = labels.to(device)

    outputs = CD_ResNet34(images)

    _, preds = torch.max(outputs, 1)

    show_images(images, labels, preds)
# show preds without any predictions

show_preds()
# training loop



def train(epochs):

    

    

    print('Starting training..')

    for e in range(0, epochs):

        print('='*20)

        print(f'Starting epoch {e + 1}/{epochs}')

        print('='*20)

        

        train_loss = 0.

        val_loss = 0.

        CD_ResNet34.train() # set model to training phase

        

        validation_loss = []

        acc = []

        

        for train_step, (images, labels) in enumerate(dl_train):

            

            

            

            images = images.to(device)

            labels = labels.to(device)

            

            optimizer.zero_grad()

            

            outputs = CD_ResNet34(images)

            loss = loss_fn(outputs, labels)

            

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            

            del labels, images

            torch.cuda.empty_cache()

            

            if train_step % 10 == 0:

                print('Evaluating at step', train_step)

                accuracy = 0



                CD_ResNet34.eval() # set model to eval phase



                for val_step, (images, labels) in enumerate(dl_test):

                    

                    images = images.to(device)

                    labels = labels.to(device)

                    

                    outputs = CD_ResNet34(images)

                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()

                    





                    _, preds = torch.max(outputs, 1)

                    

                    accuracy += sum((preds == labels).cpu().numpy())

                    

                    del labels, images

                    torch.cuda.empty_cache()



                val_loss /= (val_step + 1)

                validation_loss.append(val_loss)

                accuracy = accuracy/len(test_dataset)

                acc.append(accuracy)



                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                

                plt.subplot(1,2,1)

                plt.plot(validation_loss)

                plt.title("Validation Loss")

                plt.subplot(1,2,2)

                plt.plot(acc)

                plt.title("Accuracy")

                plt.tight_layout()

                

                

                show_preds()



                CD_ResNet34.train()



                if accuracy >= 0.99:

                    print('Performance condition satisfied, stopping..')

                    return



        train_loss /= (train_step + 1)



        print(f'Training Loss: {train_loss:.4f}')

    print('Training complete..')
%%time



train(epochs=2)
#torch.save(CD_ResNet34.state_dict(), 'CD_ResNet34_99.pth')