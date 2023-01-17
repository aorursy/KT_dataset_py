# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt



import torch

from torchvision import datasets, transforms, models



# Any results you write to the current directory are saved as output.



from torch import nn, optim

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from skimage import io, transform

import torch.utils.data as data_utils
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax
class DogBreedsDataset(Dataset):

    """Dog Breeds dataset."""



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.labels_frame = pd.read_csv(csv_file)

        self.map = dict(zip(self.labels_frame['breed'].unique(),range(0,len(self.labels_frame['breed'].unique()))))

        self.labels_frame['breed'] = self.labels_frame['breed'].map(self.map)

        self.root_dir = root_dir

        self.transform = transform

        

    def getmap(self):

        return self.map

        

    def __getclasses__(self):

        return self.labels_frame['breed'].unique().tolist()



    def __len__(self):

        return len(self.labels_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,

                                self.labels_frame.iloc[idx, 0])

        img_name = img_name + '.jpg'

        

        image = io.imread(img_name)

        PIL_image = Image.fromarray(image)

        label = self.labels_frame.iloc[idx, 1:]

        label = [int(label) for x in label]

        label = np.asarray(label)

        label = torch.from_numpy(label)

        if self.transform:

            image = self.transform(PIL_image)

        #sample = {'image': image, 'label': label}

        return image,label
class DogBreedsTestset(Dataset):

    """Dog Breeds Test dataset."""



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.labels_frame = pd.read_csv(csv_file)

        self.labels_frame = self.labels_frame[['id']]

        self.root_dir = root_dir

        self.transform = transform

   

    def __len__(self):

        return len(self.labels_frame)





    def __getitem__(self, idx):

        title = self.labels_frame.iloc[idx, 0]

        img_name = os.path.join(self.root_dir,

                                title)

        img_name = img_name + '.jpg'

        

        image = io.imread(img_name)

        PIL_image = Image.fromarray(image)

        

        if self.transform:

            image = self.transform(PIL_image)

        sample = {'image': image, 'title': title}

        return sample
data_dir = '../input'



# how many samples per batch to load

batch_size = 20

# percentage of training set to use as validation

valid_size = 0.2





# TODO: Define transforms for the training data and testing data

transform = transforms.Compose([transforms.Resize(255),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])



test_transforms = transforms.Compose([

                                      transforms.ToTensor()])



train_data = DogBreedsDataset(csv_file='../input/labels.csv',root_dir='../input/train', transform=transform)

#test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

classes = train_data.__getclasses__()

print(classes)

#obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# prepare data loaders

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

    sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

    sampler=valid_sampler)

#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
df_test = pd.read_csv('../input/sample_submission.csv')

df_test.head(1)
test_data = DogBreedsTestset(csv_file='../input/sample_submission.csv',root_dir='../input/test', transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=20)
data_iter = iter(train_loader)

images, labels = data_iter.next()

images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    #ax.set_title(classes[labels[idx]])
# Load the pretrained model from pytorch

vgg16 = models.vgg16(pretrained=True)



# print out the model structure

print(vgg16)


# Freeze training for all "features" layers

for param in vgg16.features.parameters():

    param.requires_grad = False
n_inputs = vgg16.classifier[6].in_features



# add last linear layer (n_inputs -> 5 flower classes)

# new layers automatically have requires_grad = True

last_layer = nn.Linear(n_inputs, len(classes))



vgg16.classifier[6] = last_layer



# if GPU is available, move the model to GPU

if train_on_gpu:

    vgg16.cuda()



# check to see that your last layer produces the expected number of outputs

print(vgg16.classifier[6].out_features)

#print(vgg16)
import torch.optim as optim



# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.001

optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 15



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    ###################

    # train the model #

    ###################

    # model by default is set to train

    

    for batch_i, (data, target) in enumerate(train_loader):

        # move tensors to GPU if CUDA is available

       

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = vgg16(data)

        # calculate the batch loss

        loss = criterion(output, torch.max(target, 1)[1])

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss 

        train_loss += loss.item()

        

        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches

            print('Epoch %d, Batch %d loss: %.16f' %

                  (epoch, batch_i + 1, train_loss / 20))

            train_loss = 0.0
valid_loss = 0.0

vgg16.eval()

for batch_i, (data, target) in enumerate(valid_loader):

        # move tensors to GPU if CUDA is available

       

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

            

        output = vgg16(data)

        # calculate the batch loss

        loss = criterion(output, torch.max(target, 1)[1])

        

        # update training loss 

        valid_loss += loss.item()

        

        if batch_i % 20 == 19:    # print validation loss every specified number of mini-batches

            print('Validation Loss Batch %d loss: %.16f' %

                  (batch_i + 1, valid_loss / 20))

            valid_loss = 0.0
results = {}

vgg16.eval()



for (_,data) in enumerate(test_loader):

        # move tensors to GPU if CUDA is available

        images,titles = data['image'], data['title']

               

        if train_on_gpu:

            images = images.cuda()

        #print(title)

        logits = vgg16(images)

        output = torch.nn.functional.softmax(logits, dim=1)

        

        for k in range(len(titles)):

            name = titles[k]

            results[name] = output[k].cpu().tolist()
output_df = pd.DataFrame(results).transpose()
inv_map = {v: k for k, v in train_data.getmap().items()}

inv_map
output_df.rename(columns=inv_map,inplace=True)

output_df = output_df.reset_index()

output_df.rename(columns={'index':'id'},inplace=True)
output_df.to_csv('output.csv',index=False)