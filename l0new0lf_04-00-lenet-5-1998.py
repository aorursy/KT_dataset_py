# import libs

import torch

import torch.nn as nn

import matplotlib.pyplot as plt 

import numpy as np
torch.manual_seed(123)

torch.cuda.manual_seed(123)

np.random.seed(123)

# use gpu for similar results

# not cpu
# Lenet Model

class LeNet(nn.Module):



  def __init__(self):

    super(LeNet, self).__init__()



    # activation unit

    self.relu = nn.ReLU()



    # subsampling unit(s) (independant of input, used multiple times)

    self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2)) #, padding = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: bool = None)



    # conv units

    self.conv1 = nn.Conv2d(in_channels = 1,   out_channels = 6,    kernel_size = (5,5),   stride = (1,1), padding = (0,0)) #dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')

    self.conv2 = nn.Conv2d(in_channels = 6,   out_channels = 16,   kernel_size = (5,5),   stride = (1,1), padding = (0,0))

    self.conv3 = nn.Conv2d(in_channels = 16,  out_channels = 120,  kernel_size = (5,5),   stride = (1,1), padding = (0,0))



    # overwrite self.conv1 if input img size is < 32 (comment this)

    # self.conv1 = nn.Conv2d(in_channels = 1,   out_channels = 6,    kernel_size = (5,5),   stride = (1,1), padding = (12,12)) 



    # f.c dense lin units (input_units, output_units)

    # input  -- row vec -- (1 , input_uints)

    # output -- row vec -- (1 , output_units)

    self.lin1 = nn.Linear(120, 84)

    self.lin2 = nn.Linear(84,  10)

  

  # for forward propagation

  def forward(self, x):

    

    x = self.relu( self.conv1(x) )    # 1x32x32 >> 6x28x28

    x = self.pool(x)                  # 6x28x28 >> 6x14x14



    x = self.relu( self.conv2(x) )    # 6x14x14  >> 16x10x10

    x = self.pool(x)                  # 16x10x10 >> 16x5x5



    x = self.relu( self.conv3(x) )    # 16x5x5 >> 120x1x1 (120 feature representations of same image. Each representation is 1x1 scalar)



    # flatten: Now, x is of shape (120,1,1) cz. input was (1,32,32). But,

    # While training another dimension will be added i.e

    # input will be             -- (num_samples, 1, 32, 32)

    # then, shape of x will be  -- (num_samples, 120, 1, 1)

    # So, flatten into (num_samples, -1)

    # If, single image i.e (1,32,32)   => (120,1,1)   => generates error.

    # If, single image i.e (1,1,32,32) => (1,120,1,1) => generates row vec (1,120)

    x = x.reshape(x.shape[0], -1)



    x = self.relu( self.lin1(x) )

    x = self.relu( self.lin2(x) ) # NOTE: NO SOFTMAX!!



    return x
# test

num_samples = 100

x = torch.randn(num_samples, 1, 32, 32)



model = LeNet()

print(model(x).shape)
# TEST PADDING

# ------------

import torch

import torch.nn.functional as F

source = torch.rand((5,5))

print(source.shape)



# (left_pad, right_pad, top_pad, bottom_pad)

result = F.pad(input=source, pad=(2, 1, 1, 2), mode='constant', value=0)



print(result.shape)



plt.imshow(result)
# Imports

# ---------

import torch

import torchvision

import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.

import torch.nn.functional as F # All functions that don't have any parameters

from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches

import torchvision.datasets as datasets # Has standard datasets we can import in a nice way

import torchvision.transforms as transforms # Transformations we can perform on our dataset



# Create Fully Connected Network

# ------------------------------

# done above. See `LeNet`

"""

class NN(nn.Module):

    def __init__(self, input_size, num_classes):

        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 50)

        self.fc2 = nn.Linear(50, num_classes)

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

"""



# Set device

# ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Hyperparameters

# ---------------

num_classes = 10 

learning_rate = 0.1

batch_size = 64

num_epochs = 10



# Load Data

# ---------

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# Initialize network

# ------------------

model = LeNet().to(device)#NN(input_size=input_size, num_classes=num_classes).to(device)



# Loss and optimizer

# ------------------

criterion = nn.CrossEntropyLoss() 

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Interestingly, Adam optimizer doesn't work even for large num

# of epochs !!!

optimizer = optim.SGD(model.parameters(), lr=learning_rate)



# Train Network

# -------------

for epoch in range(num_epochs):

    num_correct = 0

    num_samples = 0

    for batch_idx, (data, targets) in enumerate(train_loader):

        # Get data to cuda if possible

        data = data.to(device=device)

        targets = targets.to(device=device)

        

        # Get to correct shape

        # (sample_size, 1, 28, 28) >> (sample_size, 1, 32, 32)

        # by padding (better do it w/ matrix opn.)

        sample_size, n_channels, _, _ = data.shape

        new_data_acc = torch.rand((sample_size, n_channels, 32, 32))

        for sample_idx in range(0, sample_size):

          img = data[sample_idx, n_channels-1]

          img = F.pad(input=img, pad=(2, 2, 2, 2), mode='constant', value=0)

          # store padded images

          new_data_acc[sample_idx, n_channels-1] = img

        data = new_data_acc

        

        # forward

        scores = model(torch.tensor(data).to(device))

        loss = criterion(scores, targets)



        # log

        _, predictions = scores.max(1)

        num_correct += (targets == predictions).sum()

        num_samples += predictions.size(0)

        

        # backward

        optimizer.zero_grad()

        loss.backward()

        

        # gradient descent or adam step

        optimizer.step()



    # log

    print(f'Epoch: {epoch} >>> Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):

    if loader.dataset.train:

        print("Checking accuracy on training data")

    else:

        print("Checking accuracy on test data")

        

    num_correct = 0

    num_samples = 0

    model.eval()

    

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device)

            y = y.to(device=device)

            

            # pading 28,28 >> 32,32

            sample_size, n_channels, _, _ = x.shape

            new_data_acc = torch.rand((sample_size, n_channels, 32, 32)).to(device)

            for sample_idx in range(0, sample_size):

              img = x[sample_idx, n_channels-1]

              img = F.pad(input=img, pad=(2, 2, 2, 2), mode='constant', value=0)              

              new_data_acc[sample_idx, n_channels-1] = img

            x = new_data_acc

            

            scores = model(x)

            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()

            num_samples += predictions.size(0)

        

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 

    

    model.train()



check_accuracy(train_loader, model)

check_accuracy(test_loader, model)
import numpy as np



type(model.conv1.weight)

print(model.conv1.weight.shape)

print('total kernels = 6*1 = ', 6*1)



np_arr =  model.conv1.weight.cpu().detach().numpy()



# kernel number

i = 0



print("With min-max scaling")

plt.figure(figsize=(20,5))

for i in range(0,6):

  plt.subplot(int(f"16{i+1}"))

  plt.imshow(np_arr[i][0])



plt.show();
np_arr =  model.conv2.weight.cpu().detach().numpy()

print('conv2.weight.shape: ', np_arr.shape)

print('Total Kernels = 16x6 =', 16*6)



# Total kernels = 16*6



# kernel number (0 to 6) -- j

for j in range(0,6):

  

  # rest 8 + 8

  plt.close();

  plt.figure(figsize=(15,5))

  for i in range(0,8):

    plt.subplot(int(f"28{i+1}"))

    plt.imshow(np_arr[i][j])



  plt.show();





  plt.close();

  plt.figure(figsize=(15,5))

  for i in range(9,17):

    id = int(f"18{i-8}")

    plt.subplot(id)

    plt.imshow(np_arr[i-2][j])



  plt.show();
# vis for specific input

print("shape of raw data at random idx 4: ", train_dataset.data[4].shape)

plt.close()

plt.imshow(train_dataset.data[4])

plt.show()



# preprocess into (1,1,32,32) so that model accepts the input

preprocessed = torch.rand((1, 1, 32, 32)).to(device)



img = train_dataset.data[4] # (28,28)

img = F.pad(input=img, pad=(2, 2, 2, 2), mode='constant', value=0) #(32,32)        

preprocessed[0,0] = img



print(preprocessed.shape)



# feed specific preprocessed input to model

scores = model(preprocessed)

print(scores.shape, "\nSCORES: ", scores)

_, pred = scores.max(1)

print("PREDICTED: ", pred)
# accumulator for activations (hmap)

activation = {}



# define forward hook function

def get_activation(name):

    def hook(model, input, output):

        activation[name] = output.detach()

    return hook



# register forward hook

model.conv1.register_forward_hook(get_activation('conv1'))

model.conv2.register_forward_hook(get_activation('conv2'))



print(preprocessed.shape)

output = model(preprocessed)



print('conv1 activations size: ', activation['conv1'].shape) # 1, 6, 28x28 activation images

print('conv2 activations size: ', activation['conv2'].shape) # 1, 16, 10x10 activation images
def minmax_scale(x):

  return (x - x.min()) / (x.max()-x.min())



conv1_acts = activation['conv1']



fig, axarr = plt.subplots(1, 6, figsize=(15,5))

for act_idx in range(0,6):

  axarr[act_idx].imshow(minmax_scale(conv1_acts[0][act_idx].cpu()))
# first image as a tensor

a = minmax_scale( conv1_acts[0][0].cpu() )

print('Output of conv1\n', a)
conv2_acts = activation['conv2'] #[1, 16, 10, 10]



plt.close()

fig, axarr = plt.subplots(2, 8, figsize=(15,5))

i = -1

for act_idx_row in range(0,2):

  for act_idx_col in range(0,8):

    i=i+1

    axarr[act_idx_row, act_idx_col].imshow(minmax_scale(conv2_acts[0][i].cpu()))

plt.show()
# todo: reduce training params with distillation / pruning