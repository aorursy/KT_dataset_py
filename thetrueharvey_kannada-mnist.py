import os

import shutil

import numpy  as np

import pandas as pd



import torch

import torch.nn            as nn

import torch.nn.functional as F



from PIL         import Image

from torchvision import transforms



torch.cuda.is_available()
from torch.cuda import get_device_properties, get_device_capability, get_device_name, is_available



# GPU availability

print("Is a GPU available? {}".format(is_available()))



# Device name

print("The name of the device is {}".format(get_device_name()))



# CUDA Version

print("Device capability is {}".format(get_device_capability(0)))



# Check available GPU resources

print("Available GPU memory is {0:.{1}f} GB ".format(get_device_properties(0).total_memory / (1024 ** 3), 1))
#%% Loading

# Load the dataset

df = pd.read_csv("../input/Kannada-MNIST/test.csv")



# Create a folder

os.makedirs("images")



#%% Preprocessing

# To start, there will be no preprocessing as this is very similar to the classic MNIST dataset

# Possible preprocessing could be to remove empty space around the digits

# We could also alter pixel values to remove noise



# Bounding Box

def bbox(image):

    """

    Determines the bounding boxes for images to remove empty space where possible

    :param image:

    :return:

    """

    HEIGHT = image.shape[0]

    WIDTH  = image.shape[1]



    for i in range(image.shape[1]):

        if (image[:, i] > 0).sum() >= 1:

            x_min = i - 1 if (i > 1) else 0

            break



    for i in reversed(range(image.shape[1])):

        if (image[:, i] > 0).sum() >= 1:

            x_max = i + 2 if (i < WIDTH - 2) else WIDTH

            break



    for i in range(image.shape[0]):

        if (image[i] > 0).sum() >= 1:

            y_min = i - 1 if (i > 1) else 0

            break



    for i in reversed(range(image.shape[0])):

        if (image[i] > 0).sum() >= 1:

            y_max = i + 2 if (i < HEIGHT - 2) else HEIGHT

            break



    return x_min, y_min, x_max, y_max



#%% Dataset generation

# Label and image ID storage

ids    = []

labels = []



# Loop through each row

for i,row in df.iterrows():

    # Get the data components

    id    = "{}".format(i)

    img   = np.reshape(row[1:].values, newshape=(28,28)).astype(np.uint8)



    # Remove empty space

    x_min, y_min, x_max, y_max = bbox(img)

    img = img[y_min:y_max, x_min:x_max]



    # Check again

    x_min, y_min, x_max, y_max = bbox(img)

    img = img[y_min:y_max, x_min:x_max]



    # Pad

    img = np.pad(img, (2,2), "constant", constant_values=(0,0))



    # Remove any non-maximum pixels

    img[img >= 50] = 255

    img[img < 50] = 0



    # Convert and reshape

    img = Image.fromarray(img)

    img = img.resize((28,28), Image.ANTIALIAS)



    # Fetch the label

    label = 0



    # Write to storage

    img.save("images/{}.png".format(id))

    ids.append(id)

    labels.append(label)

    

# Save the IDs to a DataFrame and write to CSV

label_df = pd.DataFrame({"id": ids, "label": labels})

class KannadaMNIST(object):

    def __init__( self

                , image_folder    : str

                , labels          : pd.DataFrame

                , transforms      : list  = None

                ):

        """

        Class for creating a dataset for training the flow chart object detector

        """

        # Class attributes

        self.image_folder = image_folder

        self.labels       = labels

        self.transforms   = transforms



        # Create an index list

        self.index = np.arange(self.labels.shape[0])



    def __len__(self):

        return len(self.index)



    def __getitem__(self, idx):

        # Load the labels

        label = self.labels.iloc[self.index[idx]]



        # Load the image

        img = Image.open(self.image_folder + "/" + label["id"] + ".png")



        # Apply transformations

        img_ = self.transforms(img)



        return {"image": img_, "id":int(label["id"])}
import torch.nn as nn

import torch, math, sys



####

# CODE TAKEN FROM https://github.com/sdoria/SimpleSelfAttention

####



#Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py

def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):

	"Create and initialize a `nn.Conv1d` layer with spectral normalization."

	conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)

	nn.init.kaiming_normal_(conv.weight)

	if bias: conv.bias.data.zero_()

	return nn.utils.spectral_norm(conv)



# Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py

# Inspired by https://arxiv.org/pdf/1805.08318.pdf

class SimpleSelfAttention(nn.Module):

	

	def __init__(self, n_in, ks=1, sym=False):

		super().__init__()        

		self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)            

		self.gamma = nn.Parameter(torch.Tensor([0.]))      

		self.sym = sym

		self.n_in = n_in

		

	def forward(self, x):

		if self.sym:

			# symmetry hack by https://github.com/mgrankin

			c = self.conv.weight.view(self.n_in,self.n_in)

			c = (c + c.t())/2

			self.conv.weight = c.view(self.n_in,self.n_in,1)

				

		size = x.size()  

		x = x.view(*size[:2],-1)   # (C,N)

		

		# changed the order of mutiplication to avoid O(N^2) complexity

		# (x*xT)*(W*x) instead of (x*(xT*(W*x)))

		

		convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)

		xxT = torch.bmm(x, x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)  		    

		o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)         

		o = self.gamma * o + x  

		  

		return o.view(*size).contiguous()  
"""

Network for Kannada MNIST classification

"""

# %% Setup

import torch

import torch.nn            as nn

import torch.nn.functional as F





# %% Network

class KannadaNet(nn.Module):

    def __init__(self

                 , input_shape

                 , n_class

                 ):

        """

        Builds a model for the Kannada MNIST dataset

        Based on: https://www.kaggle.com/bustam/cnn-in-keras-for-kannada-digits

        :param input_shape:

        :param n_class:

        """

        super(KannadaNet, self).__init__()



        # Build the layers

        self._build_layers(input_shape=input_shape, n_class=n_class)



        # Initialize weights

        self._init_weight()



        # Parameter counts

        print("Model contains {} parameters".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))



    def forward(self, x):

        x = self._conv_forward(x)



        # Classification

        x = self.fc_1(x)

        x = F.relu(x)

        x = self.bn_1(x)



        x = self.fc_2(x)

        x = F.relu(x)

        x = self.bn_2(x)



        confidence = self.confidence(x)

        out        = self.classifier(x)



        #return out

        return {"prediction":out.cpu().float(), "confidence":confidence.cpu().float()}



    def _build_layers(self, input_shape, n_class):

        """

        Adds layers to the network

        :param input_shape:

        :param n_class:

        :return:

        """

        # Create the stem of the network

        self.stem = Conv2D_BN( in_channels =input_shape[0]

                             , out_channels=32

                             , padding     =1

                             )



        # Inception blocks

        self.inception_res_1 = InceptionBlock(in_shape=[32,28,28], residual=True)

        self.attention_1 = SimpleSelfAttention(self.inception_res_1.output_shape[0])

        self.inception_res_2 = InceptionBlock(in_shape=[int(x / y) for x, y in zip(self.inception_res_1.output_shape, [1,2,2])], residual=True)

        self.attention_2 = SimpleSelfAttention(self.inception_res_2.output_shape[0])

        self.inception_res_3 = InceptionBlock(in_shape=[int(x / y) for x, y in zip(self.inception_res_2.output_shape, [1, 2, 2])], residual=True)

        self.attention_3 = SimpleSelfAttention(self.inception_res_3.output_shape[0])



        # Pooling

        self.pool = nn.MaxPool2d(kernel_size=2)



        # Determine the output dimensions

        conv_out_dims = self._conv_forward(torch.zeros([1] + input_shape, dtype=torch.float32))



        # Linear layers

        self.fc_1 = nn.Linear(in_features=conv_out_dims.shape[1], out_features=512)

        self.bn_1 = nn.BatchNorm1d(num_features=512)

        self.fc_2 = nn.Linear(in_features=512, out_features=512)

        self.bn_2 = nn.BatchNorm1d(num_features=512)



        self.confidence = nn.Linear(in_features=512, out_features=1)

        self.classifier = nn.Linear(in_features=512, out_features=n_class)





    # Convolutional forward

    def _conv_forward(self, x):

        """

        Forward function for convolutional layers

        :param x:

        :return:

        """

        x = self.stem(x)

        x = self.inception_res_1(x)

        x = self.attention_1(x)

        x = self.pool(x)

        x = self.inception_res_2(x)

        x = self.attention_2(x)

        x = self.pool(x)

        x = self.inception_res_3(x)

        x = self.attention_3(x)

        x = self.pool(x)



        # Flatten

        x = torch.flatten(x, start_dim=1)



        return x



    # Weight initialization

    def _init_weight(self):

        for m in self.modules():

            print(m)

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1.0)

                m.bias.data.zero_()



# %% Utility functions

class InceptionBlock(nn.Module):

    def __init__(self, in_shape, residual=True):

        super(InceptionBlock, self).__init__()

        # Class parameters

        self.residual = residual



        # Define the component operations

        self.conv_1x1_1 = nn.Conv2d(kernel_size=1, in_channels=in_shape[0], out_channels=int(in_shape[0]/2))

        self.bn_1       = nn.BatchNorm2d(num_features=self.conv_1x1_1.out_channels)



        self.conv_1x1_2 = nn.Conv2d(kernel_size=1, in_channels=in_shape[0], out_channels=int(in_shape[0]))

        self.bn_2       = nn.BatchNorm2d(num_features=self.conv_1x1_2.out_channels)



        self.conv_1x1_3 = nn.Conv2d(kernel_size=1, in_channels=in_shape[0], out_channels=int(in_shape[0]/2))

        self.bn_3       = nn.BatchNorm2d(num_features=self.conv_1x1_3.out_channels)



        self.conv_1x1_4 = nn.Conv2d(kernel_size=1, in_channels=in_shape[0], out_channels=int(in_shape[0]/2))

        self.bn_4       = nn.BatchNorm2d(num_features=self.conv_1x1_4.out_channels)





        self.conv_3x3_1 = nn.Conv2d(kernel_size=3, in_channels=self.conv_1x1_2.out_channels, out_channels=self.conv_1x1_1.out_channels * 2, padding=1)

        self.bn_5       = nn.BatchNorm2d(num_features=self.conv_3x3_1.out_channels)



        self.conv_3x3_2 = nn.Conv2d(kernel_size=3, in_channels=self.conv_1x1_3.out_channels, out_channels=int(in_shape[0]/4), padding=1)

        self.bn_6       = nn.BatchNorm2d(num_features=self.conv_3x3_2.out_channels)



        self.conv_3x3_3 = nn.Conv2d(kernel_size=3, in_channels=self.conv_3x3_2.out_channels, out_channels=int(in_shape[0]/4), padding=1)

        self.bn_7       = nn.BatchNorm2d(num_features=self.conv_3x3_3.out_channels)



        self.pool_3x3   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)



        # Determine the block output shape

        self.output_shape = self._get_out_shape(in_shape)



    def forward(self, x):

        # First path

        x_1 = self.conv_1x1_1(x)

        x_1 = F.relu(x_1)

        x_1 = self.bn_1(x_1)





        # Second path

        x_2 = self.conv_1x1_2(x)

        x_2 = F.relu(x_2)

        x_2 = self.bn_2(x_2)



        x_2 = self.conv_3x3_1(x_2)

        x_2 = F.relu(x_2)

        x_2 = self.bn_5(x_2)



        

        # Third path

        x_3 = self.conv_1x1_3(x)

        x_3 = F.relu(x_3)

        x_3 = self.bn_3(x_3)



        x_3 = self.conv_3x3_2(x_3)

        x_3 = F.relu(x_3)

        x_3 = self.bn_6(x_3)



        x_3 = self.conv_3x3_3(x_3)

        x_3 = F.relu(x_3)

        x_3 = self.bn_7(x_3)



        

        # Fourth path

        x_4 = self.pool_3x3(x)

        x_4 = self.conv_1x1_4(x_4)

        x_4 = F.relu(x_4)

        x_4 = self.bn_4(x_4)





        # Concatenate

        if self.residual:

            x = torch.cat([x_1, x_2, x_3, x_4, x], dim=1)

        else:

            x = torch.cat([x_1, x_2, x_3, x_4], dim=1)



        return x



    def _get_out_shape(self, input_shape):

        return list(self.forward(torch.zeros([1] + input_shape)).shape[1:])





class Conv2D_BN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, activation=True):

        """

        Convolution followed by Batch Normalization, with optional Mish activation

        :param in_features:

        :param out_features:

        :param kernel_size:

        :param stride:

        :param padding:

        :param groups:

        :param activation:

        """

        super(Conv2D_BN, self).__init__()



        # Class parameters

        self.activation = activation



        # Convolution and BatchNorm class

        self.conv = nn.Conv2d( in_channels =in_channels

                             , out_channels=out_channels

                             , kernel_size =kernel

                             , stride      =stride

                             , padding     =padding

                             , bias        =True

                             )



        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)



    def forward(self, x):

        x = self.conv(x)

        if self.activation:

            x = F.relu(x)

        x = self.bn(x)





        return x





def block_forward(layers, x):

    for layer in layers:

        x = layer(x)



    return x





# Mish Activation

def Mish(x):

    r"""

    Mish activation function is proposed in "Mish: A Self

    Regularized Non-Monotonic Neural Activation Function"

    paper, https://arxiv.org/abs/1908.08681.

    """



    return x * torch.tanh(F.softplus(x))

# Create the dataset

dataset = KannadaMNIST(image_folder="images", labels=label_df, transforms=transforms.ToTensor())



# Create a dataloader

loader = torch.utils.data.DataLoader( dataset    =dataset

                                    , batch_size =256

                                    , shuffle    =False

                                    #, num_workers=4

                                    , pin_memory =True

                                    )
# Check the loader

for sample in loader:

    test = sample["image"][0]

    break

    

test = transforms.functional.to_pil_image(test)

test
# Instantiate and load a model

net = KannadaNet( input_shape=[1, 28, 28]

                , n_class    =10

                )

net.load_state_dict(torch.load("../input/pytorch-kannada-v2/CustomNetV2 Submission Weights.tar"), strict=True)

net.cuda()

net.eval()
# Inference

predictions = []

ids         = []

with torch.no_grad():

    for x in loader:

        predictions.append(F.softmax(net(x["image"].cuda())["prediction"], dim=1).cpu().argmax(dim=1))

        ids.append(x["id"])



predictions = torch.cat(predictions)

ids         = torch.cat(ids)
ids
predictions.shape
# Load the sample and update the predictions

label_df['label'] = predictions

label_df['id']    = ids



label_df = label_df.sort_values("id")



# Check the submission

print(label_df.head())



# Write the submission

label_df.to_csv("submission.csv",index=False)
# Clean up

shutil.rmtree("images")