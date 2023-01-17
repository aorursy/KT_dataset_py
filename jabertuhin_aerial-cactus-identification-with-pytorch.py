import numpy as np

import pandas as pd 

import cv2

from matplotlib import pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



# to unzip

import zipfile



# Import PyTorch

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

import torch.optim as optim
# ref: https://stackoverflow.com/questions/3451111/unzipping-files-in-python/3451150

def unzip(path):

    with zipfile.ZipFile(path,"r") as z:

        z.extractall('.')
# unzip train folder

train_zip_path = '../input/train.zip'

unzip(train_zip_path)
test_zip_path = '../input/test.zip'

unzip(test_zip_path)
train_df = pd.read_csv('../input/train.csv')

train_df.head()
def draw_pie_chart(labels, sizes, explode=None):        

    fig, ax = plt.subplots()

    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    plt.show()    
number_of_images_contains_cactus = train_df.has_cactus.value_counts().loc[1]

number_of_images_does_not_contain_cactus = train_df.has_cactus.value_counts().loc[0]



print(number_of_images_contains_cactus)

print(number_of_images_does_not_contain_cactus)



draw_pie_chart(["has_cactus", "no_cactus"], [number_of_images_contains_cactus, number_of_images_does_not_contain_cactus], (0, 0.1))
number_of_training_images = len(os.listdir('/kaggle/working/train'))

number_of_test_images = len(os.listdir('/kaggle/working/test'))



print(f"Total Train Images: {number_of_training_images}")

print(f"Total Test Images: {number_of_test_images}")



draw_pie_chart(["Total Training Images", "Total Test Images"], [number_of_training_images, number_of_test_images], (0, 0.1))
class Dataset(Dataset):

    def __init__(self, dataframe, data_dir, transform=None):

        super().__init__()

        self.dataframe = dataframe

        self.data_dir = data_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self, index):

        image_name, label = self.dataframe.iloc[index]

        image_path = os.path.join(self.data_dir, image_name)

        image = cv2.imread(image_path)

        

        if self.transform != None:

            image = self.transform(image)

        

        return image, label            
transforms_train_data = transforms.Compose([transforms.ToPILImage(),

                                      transforms.RandomHorizontalFlip(),

                                      transforms.RandomRotation(10),

                                      transforms.ToTensor(),

                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transforms_train_data
train_data = Dataset(train_df, '/kaggle/working/train', transforms_train_data)
batch_size = 64



validation_size = int(np.floor(0.2*number_of_training_images))



indices = list(range(number_of_training_images))

np.random.shuffle(indices)



train_indices, validation_indices = indices[validation_size:], indices[:validation_size]



train_sampler = SubsetRandomSampler(train_indices)

validation_sampler = SubsetRandomSampler(validation_indices)



train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

validation_loader = DataLoader(train_data, batch_size=batch_size, sampler=validation_sampler)
sample_submission_path = "../input/sample_submission.csv"

sample_submission_df = pd.read_csv(sample_submission_path)

sample_submission_df.head()
transform_test_data = transforms.Compose([transforms.ToPILImage(),

                                     transforms.ToTensor(),

                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test_data
test_data = Dataset(sample_submission_df, '/kaggle/working/test', transform_test_data)



test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
classes  = ['Cactus', 'No Cactus']
def show_image(image):

    '''Helper function to un-normalize and display an image'''

    # unnormalize

    image = image / 2 + 0.5

    # convert from Tensor image and display

    plt.imshow(np.transpose(image, (1, 2, 0)))
data_iter = iter(train_loader)

images, labels = data_iter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(10):

    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])

    show_image(images[idx])

    print(images[idx].shape)

    ax.set_title(classes[labels[idx]])
class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Convolutional Layer (sees 32x32x3 image tensor | outputs 16x16x16 image tensor) 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Convolutional Layer (sees 16x16x16 image tensor | outputs 8x8x32 image tensor)  

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Convolutional Layer (sees 8x8x32 image tensor | outputs 4x4x64 image tensor)  

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Convolutional Layer (sees 4x4x64 image tensor | outputs 2x2x128 image tensor)  

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)        

        

        # batch normalization

        self.bn1 = nn.BatchNorm2d(16)

        self.bn2 = nn.BatchNorm2d(32)

        self.bn3 = nn.BatchNorm2d(64)

        self.bn4 = nn.BatchNorm2d(128)        

        

        # MaxPooling Layer

        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers

        self.fc1 = nn.Linear(2*2*128, 512)        

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 2)

        

        self.dropout = nn.Dropout(0.2)

    

    def forward(self, x):        

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool(x)

        

        x = x.view(-1, 2*2*128)

        x = self.dropout(x)

        

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        

        x = F.relu(self.fc2(x))                        

        x = self.dropout(x)

        

        x = F.relu(self.fc3(x))

        

        return x
class Model:

    def __init__(self, model, criterion, optimizer, train_on_gpu, train_loader, validation_loader):

        self.model = model        

        self.criterion = criterion

        self.optimizer = optimizer

        self.train_on_gpu = train_on_gpu                        

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.train_losses = []

        self.validation_losses = []        

        if self.train_on_gpu:

            self.model.cuda()

            

    def fit(self, epochs=30, show_every=1):

        self.train_losses = []

        self.validation_losses = []

        

        minimum_validation_loss = np.inf        

        for epoch in range(epochs):                               

            train_loss = self.train()

            self.train_losses.append(train_loss)

                        

            validation_loss = self.validate()

            self.validation_losses.append(validation_loss)   

            

            if epoch%show_every == 0:

                self.print_loss(epoch, train_loss, validation_loss)



            if validation_loss < minimum_validation_loss: 

                self.save_model(minimum_validation_loss, validation_loss)

                minimum_validation_loss = validation_loss        

    

    def train(self):

        train_loss = 0.0



        self.model.train()

        for data, target in self.train_loader:

            if self.train_on_gpu:

                data, target = data.cuda(), target.cuda()



            self.optimizer.zero_grad()



            output = self.model(data)



            loss = self.criterion(output, target)



            loss.backward()



            self.optimizer.step()



            train_loss += loss.item()*data.size(0)       





        train_loss = train_loss/len(self.train_loader.sampler)

        return train_loss

        

    

    def validate(self):                

        validation_loss = 0.0

        

        self.model.eval()

        

        for data, target in self.validation_loader:

            if self.train_on_gpu:

                data, target = data.cuda(), target.cuda()



            output = self.model(data)



            loss = self.criterion(output, target)



            validation_loss += loss.item()*data.size(0)

            

        validation_loss = validation_loss/len(self.validation_loader.sampler)

        return validation_loss

    

    def print_loss(self, epoch, train_loss, validation_loss):        

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss, validation_loss))        

    

    def save_model(self, minimum_validation_loss, validation_loss):

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(minimum_validation_loss, validation_loss))            

        torch.save(model.state_dict(), 'best_model.pt')        

    

    def load_best_model(self):

        self.model.load_state_dict(torch.load('best_model.pt'))

    

    def predict(self, test_loader):

        self.load_best_model()        

        self.model.eval()



        predictions = []



        for data, target in test_loader:

            if self.train_on_gpu:

                    data, target = data.cuda(), target.cuda()



            output = self.model(data)



            prediction = output[:,1].detach().cpu().numpy()

            for pred in prediction:

                predictions.append(pred)

                

        return predictions

    

    def show_loss_graph(self):

        plt.style.use('seaborn')

        plt.plot(self.validation_losses, label='Validation loss')

        plt.plot(self.train_losses, label='Train loss')

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.show()        
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
model = CNN()



criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters(), lr=0.001)



cnn_model = Model(model, criterion, optimizer, train_on_gpu, train_loader, validation_loader)
cnn_model.fit(50)
cnn_model.show_loss_graph()
predictions = cnn_model.predict(test_loader)
sample_submission_df["has_cactus"] = predictions

sample_submission_df["has_cactus"] = sample_submission_df["has_cactus"].apply(lambda x: 1 if x > 0.75 else 0)

sample_submission_df.to_csv("submission.csv", index=False)
import shutil





shutil.rmtree('/kaggle/working/train')

shutil.rmtree('/kaggle/working/test')