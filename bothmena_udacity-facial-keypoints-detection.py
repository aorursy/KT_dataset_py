!git clone https://github.com/udacity/P1_Facial_Keypoints.git

!mv P1_Facial_Keypoints/data/ .
!rm -rf P1_Facial_Keypoints
# import the required libraries

import glob

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import torch

from torch.autograd import Variable

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torch import nn, optim

from torch.nn import functional as F



import cv2
class FacialKeypointsDataset(Dataset):

    """Face Landmarks dataset."""



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.key_pts_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.key_pts_frame)



    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir,

                                self.key_pts_frame.iloc[idx, 0])

        

        image = mpimg.imread(image_name)

        

        # if image has an alpha color channel, get rid of it

        if(image.shape[2] == 4):

            image = image[:,:,0:3]

        

        key_pts = self.key_pts_frame.iloc[idx, 1:].values

        key_pts = key_pts.astype('float').reshape(-1, 2)

        sample = {'image': image, 'keypoints': key_pts}



        if self.transform:

            sample = self.transform(sample)



        return sample
class Normalize(object):

    """Convert a color image to grayscale and normalize the color range to [0,1]."""        



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']

        

        image_copy = np.copy(image)

        key_pts_copy = np.copy(key_pts)



        # convert image to grayscale

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        

        # scale color range from [0, 255] to [0, 1]

        image_copy=  image_copy/255.0

            

        

        # scale keypoints to be centered around 0 with a range of [-1, 1]

        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50

        key_pts_copy = (key_pts_copy - 100)/50.0





        return {'image': image_copy, 'keypoints': key_pts_copy}
class Rescale(object):

    """Rescale the image in a sample to a given size.

    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']



        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)



        img = cv2.resize(image, (new_w, new_h))

        

        # scale the pts, too

        key_pts = key_pts * [new_w / w, new_h / h]



        return {'image': img, 'keypoints': key_pts}
class RandomCrop(object):

    """Crop randomly the image in a sample.

    Args:

        output_size (tuple or int): Desired output size. If int, square crop

            is made.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']



        h, w = image.shape[:2]

        new_h, new_w = self.output_size



        top = np.random.randint(0, h - new_h)

        left = np.random.randint(0, w - new_w)



        image = image[top: top + new_h,

                      left: left + new_w]



        key_pts = key_pts - [left, top]



        return {'image': image, 'keypoints': key_pts}
class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']

         

        # if image has no grayscale color channel, add one

        if(len(image.shape) == 2):

            # add that third color dim

            image = image.reshape(image.shape[0], image.shape[1], 1)

            

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        

        return {'image': torch.from_numpy(image),

                'keypoints': torch.from_numpy(key_pts)}
data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

# create the transformed dataset

transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',

                                             root_dir='data/training/',

                                             transform=data_transform)





print('Number of images: ', len(transformed_dataset))

batch_size = 64



train_loader = DataLoader(transformed_dataset, 

                          batch_size=batch_size,

                          shuffle=True, 

                          num_workers=0)

# create the test dataset

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv', root_dir='data/test/', transform=data_transform)



# load test data in batches

test_loader = DataLoader(test_dataset,  batch_size=10, shuffle=True, num_workers=0)
def show_keypoints(image, key_pts):

    """Show image with keypoints"""

    plt.imshow(image)

    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
def net_sample_output():

    

    # iterate through the test dataset

    for i, sample in enumerate(test_loader):

        

        # get sample data: images and ground truth keypoints

        images = sample['image']

        key_pts = sample['keypoints']



        # convert images to FloatTensors

        if torch.cuda.is_available():

            images = images.type(torch.cuda.FloatTensor)

        else:

            images = images.type(torch.FloatTensor)



        # forward pass to get net output

        output_pts = net(images)

        

        # reshape to batch_size x 68 x 2 pts

        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        

        # break after first image is tested

        if i == 0:

            return images, output_pts, key_pts

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):

    """Show image with predicted keypoints"""

    # image is grayscale

    plt.imshow(image, cmap='gray')

    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')

    # plot ground truth points as green pts

    if gt_pts is not None:

        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):



    for i in range(batch_size):

        plt.figure(figsize=(20,20))

        ax = plt.subplot(1, batch_size, i+1)



        # un-transform the image data

        image = test_images[i].data   # get the image from it's Variable wrapper

        if torch.cuda.is_available():

            image = image.cpu()

        image = image.numpy()   # convert to numpy array from a Tensor

        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image



        # un-transform the predicted key_pts data

        predicted_key_pts = test_outputs[i].data

        if torch.cuda.is_available():

            predicted_key_pts = predicted_key_pts.cpu()

        predicted_key_pts = predicted_key_pts.numpy()

        # undo normalization of keypoints  

        predicted_key_pts = predicted_key_pts*50.0+100

        

        # plot ground truth points for comparison, if they exist

        ground_truth_pts = None

        if gt_pts is not None:

            ground_truth_pts = gt_pts[i]         

            ground_truth_pts = ground_truth_pts*50.0+100

        

        # call show_all_keypoints

        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

            

        plt.axis('off')



    plt.show()

criterion = nn.SmoothL1Loss()
def net_test_loss():

    avg_loss = 0.0

    with torch.no_grad():

        for i, sample in enumerate(test_loader):

        

            # get sample data: images and ground truth keypoints

            images = sample['image']

            key_pts = sample['keypoints']



            # flatten pts

            key_pts = key_pts.view(key_pts.size(0), -1)



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                key_pts = key_pts.cuda().float()

                images = images.cuda().float()

            else:

                key_pts = key_pts.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)



            # forward pass to get net output

            output_pts = net(images)

            loss = criterion(output_pts, key_pts)



            avg_loss += loss.item() / len(test_loader)

    return avg_loss

class Network(nn.Module):



    def __init__(self):

        super(Network, self).__init__()



        ## TODO: Define all the layers of this CNN, the only requirements are:

        ## 1. This network takes in a square (same width and height), grayscale image as input

        ## 2. It ends with a linear layer that represents the keypoints

        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        ## Note that among the layers to add, consider including:

        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting



        # input 1x224x224

        # (W-F)/S +1

        self.features = nn.Sequential(

            nn.Conv2d(1, 64, 3, 1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # => 64 * 112 * 112

            

            nn.Conv2d(64, 128, 3, 1, padding=1),

#             nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # => 128 * 56 * 56

            

            nn.Conv2d(128, 256, 3, 1, padding=1),

#             nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # => 256 * 28 * 28

            

            nn.Conv2d(256, 512, 3, 1, padding=1),

#             nn.BatchNorm2d(512),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # => 512 * 14 * 14

            

            nn.Conv2d(512, 512, 3, 1, padding=1),

#             nn.BatchNorm2d(512),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # => 512 * 7 * 7

        )

        output_ft = 512 * 7 * 7

        self.regressor = nn.Sequential(

            nn.Linear(output_ft, 8192),

            nn.Dropout(0.5),

            nn.ReLU(),

            nn.Linear(8192, 1024),

            nn.Dropout(0.5),

            nn.ReLU(),

            nn.Linear(1024, 136),

        )



    def forward(self, x):

        ## TODO: Define the feedforward behavior of this model

        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.regressor(x)



        return x

def weights_init(m):

    if isinstance(m, nn.Conv2d):

        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:

            m.bias.data.fill_(0.01)

    if isinstance(m, nn.Linear):

        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:

            m.bias.data.fill_(0.01)

def lr_decay(iteration, init_lr=0.001, decay=0.5, cycle=None):

    if isinstance(cycle, int):

        iteration = iteration % cycle

    return init_lr / (1 + decay * iteration)

def train_net(n_epochs, print_every=10, lr=0.001, lrd=False, lrd_cycle=None):



    # prepare the net for training

    net.train()

    if not lrd:

        optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []



    for epoch in range(n_epochs):  # loop over the dataset multiple times

        

        if lrd:

            new_lr = lr_decay(epoch, lr, cycle=lrd_cycle)

            optimizer = optim.Adam(net.parameters(), lr=new_lr)

        

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader

        for batch_i, data in enumerate(train_loader):

            # get the input images and their corresponding labels

            images = data['image']

            key_pts = data['keypoints']



            # flatten pts

            key_pts = key_pts.view(key_pts.size(0), -1)



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                key_pts = key_pts.cuda().float()

                images = images.cuda().float()

            else:

                key_pts = key_pts.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)



            # forward pass to get outputs

            output_pts = net(images)



            # calculate the loss between predicted and target keypoints

            loss = criterion(output_pts, key_pts)



            # zero the parameter (weight) gradients

            optimizer.zero_grad()

            

            # backward pass to calculate the weight gradients

            loss.backward()



            # update the weights

            optimizer.step()



            # print loss statistics

            running_loss += loss.item()

            if (batch_i + 1) % print_every == 0:

                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/print_every))

                losses.append(running_loss/print_every)

                running_loss = 0.0



    print('Finished Training')

    return losses

def save_net(name: str):

    if not os.path.isdir('models'):

        os.mkdir('models', 0o777)

    torch.save(net.state_dict(), 'models/' + name + '.pt')

experiences = {}
# train your network

n_epochs = 200 # start small, and increase when you've decided on your model structure and hyperparams



# this is a Workspaces-specific context manager to keep the connection

# alive while training your model, not part of pytorch

# for lr in [0.001, 0.0001, 0.00001]:



net = Network()

if torch.cuda.is_available():

    net.cuda()

net.apply(weights_init)

print(net)

losses = train_net(n_epochs, print_every=25, lr=0.001, lrd=True, lrd_cycle=40)

name = 'net_v4_cyclic_lrd_200'

experiences[name] = losses

save_net(name)

training_loss = np.mean(losses[-5:])

test_loss = net_test_loss()



print('Training Loss: ', training_loss)

print('Test Loss: ', test_loss)

print('Difference: ', test_loss - training_loss)
plt.figure(figsize=(20, 10))

for exp, losses in experiences.items():

    plt.plot(losses[1:], label=exp)



plt.legend()

plt.show()

plt.savefig('loss.png')

test_images, test_outputs, gt_pts = net_sample_output()

visualize_output(test_images, test_outputs, gt_pts, 5)
# get a sample of test data again

test_images, test_outputs, gt_pts = net_sample_output()



print(test_images.data.size())

print(test_outputs.data.size())

print(gt_pts.size())
## TODO: visualize your test output

# you can use the same function as before, by un-commenting the line below:



visualize_output(test_images, test_outputs, gt_pts, 5)
!rm -rf data