!pip install Pillow==6.1 # Version7でPILLOW_VERSIONが無くなった為
from PIL import Image

import matplotlib.pyplot as plt

import os

import glob

import torch

from torch import optim 

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import numpy as np
class Dataset(Dataset):



    # Constructor

    def __init__(self,transform=None,train=True):

        directory="../input/us-concrete-crack"

        positive="Positive"

        negative="Negative"



        positive_file_path=os.path.join(directory,positive)

        negative_file_path=os.path.join(directory,negative)

        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]

        positive_files.sort()

        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]

        negative_files.sort()

        number_of_samples=len(positive_files)+len(negative_files)

        self.all_files=[None]*number_of_samples

        self.all_files[::2]=positive_files

        self.all_files[1::2]=negative_files 

        # The transform is goint to be used on image

        self.transform = transform

        #torch.LongTensor

        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)

        self.Y[::2]=1

        self.Y[1::2]=0

        

        if train:

            self.all_files=self.all_files[0:30000]

            self.Y=self.Y[0:30000]

            self.len=len(self.all_files)

        else:

            self.all_files=self.all_files[30000:]

            self.Y=self.Y[30000:]

            self.len=len(self.all_files)    

       

    # Get the length

    def __len__(self):

        return self.len

    

    # Getter

    def __getitem__(self, idx):

        

        

        image=Image.open(self.all_files[idx])

        y=self.Y[idx]

          

        

        # If there is any transform method, apply it onto the image

        if self.transform:

            image = self.transform(image)



        return image, y
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

# transforms.ToTensor()

#transforms.Normalize(mean, std)

#transforms.Compose([])



transform =transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset_train=Dataset(transform=transform,train=True)

dataset_val=Dataset(transform=transform,train=False)
def show_data(data_sample, shape = (227, 227, 3)):

    # (3 x 227 x 227) => (227 x 227 x 3)

    transposed_imarr = np.transpose(data_sample[0].numpy().reshape((3, 227, 227)), (1, 2, 0))

    # BGR format => RGB format

    assert transposed_imarr.shape == shape

    B, G, R = transposed_imarr.T

    rgb_imarr = np.array((R, G, B)).T

    plt.imshow(rgb_imarr)

    plt.title('y = {}'.format(data_sample[1]))

    plt.show()



print("Shape of data element: ", dataset_train[0][0].shape)

print("Type of data element: ", dataset_train[0][1].type())

for i in range(4):

    show_data(dataset_train[i])
# Create the data loader

train_loader = DataLoader(dataset=dataset_train, batch_size=1000)



# Create the data loader

val_loader = DataLoader(dataset=dataset_val, batch_size=1000)
# Define a function to plot accuracy and loss

def plot_accuracy_loss(training_results): 

    plt.subplot(2, 1, 1)

    plt.plot(training_results['train_loss'], 'r', label='training loss')

    plt.ylabel('loss')

    plt.title('training loss iterations')

    plt.legend(loc="best")

    plt.subplot(2, 1, 2)

    plt.plot(training_results['validation_acc'], label='validation accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epochs')

    plt.legend(loc="best")

    plt.show()
# The class for plotting



class plot_diagram():

    

    # Constructor

    def __init__(self):

        self.error = []

        self.parameter = []

        

    # Executor

    def __call__(self, Yhat, w, error, n):

        self.error.append(error)

        self.parameter.append(w.data)

        plt.subplot(212)

        plt.plot(self.X, Yhat.detach().numpy())

        plt.plot(self.X, self.Y,'ro')

        plt.xlabel("A")

        plt.ylim(-20, 20)

        plt.subplot(211)

        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))

        plt.plot(self.parameter_values.numpy(), self.Loss_function)   

        plt.plot(self.parameter, self.error, 'ro')

        plt.xlabel("B")

        plt.figure()

    

    # Destructor

    def __del__(self):

        plt.close('all')
# Define the function for plotting the channels



def plot_channels(W):

    n_out = W.shape[0]

    n_in = W.shape[1]

    w_min = W.min().item()

    w_max = W.max().item()

    fig, axes = plt.subplots(n_out, n_in)

    fig.subplots_adjust(hspace=0.1)

    out_index = 0

    in_index = 0

    

    #plot outputs as rows inputs as columns 

    for ax in axes.flat:

        if in_index > n_in-1:

            out_index = out_index + 1

            in_index = 0

        ax.imshow(W[out_index, in_index, :, :], vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_yticklabels([])

        ax.set_xticklabels([])

        in_index = in_index + 1



    plt.show()
# Define the function for plotting the parameters



def plot_parameters(W, number_rows=1, name="", i=0):

    W = W.data[:, i, :, :]

    n_filters = W.shape[0]

    w_min = W.min().item()

    w_max = W.max().item()

    fig, axes = plt.subplots(number_rows, n_filters // number_rows)

    fig.subplots_adjust(hspace=0.4)



    for i, ax in enumerate(axes.flat):

        if i < n_filters:

            # Set the label for the sub-plot.

            ax.set_xlabel("kernel:{0}".format(i + 1))



            # Plot the image.

            ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])

            ax.set_yticks([])

    plt.suptitle(name, fontsize=10)    

    plt.show()
# Define the function for plotting the activations

def plot_activations(A, number_rows=1, name="", i=0):

    A = A[0, :, :, :].detach().numpy()

    n_activations = A.shape[0]

    A_min = A.min().item()

    A_max = A.max().item()

    fig, axes = plt.subplots(number_rows, n_activations // number_rows)

    fig.subplots_adjust(hspace = 0.4)



    for i, ax in enumerate(axes.flat):

        if i < n_activations:

            # Set the label for the sub-plot.

            ax.set_xlabel("activation:{0}".format(i + 1))



            # Plot the image.

            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')

            ax.set_xticks([])

            ax.set_yticks([])

    plt.show()
torch.manual_seed(0)

class Net(nn.Module):

    

    # Constructor

    def __init__(self, input_channels=3, out_1=8, out_2=16, n_class=2):

        super(Net, self).__init__()

        # x = [N, 3, 227, 227]

        self.conv1 = nn.Conv2d(

            in_channels=input_channels,

            out_channels=out_1,

            kernel_size=3,

            stride=2 # without padding

        )

        # x = [N, 8, 113, 113] ※ dimension = ((n+2*padding-filter)/stride)+1 (n=height or width)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # x = [N, 8, 56, 56]

        self.conv2 = nn.Conv2d(

            in_channels=out_1,

            out_channels=out_2,

            kernel_size=3,

            stride=2 # without padding

        )

        # x = [N, 16, 27, 27]

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # x = [N, 16, 13, 13]

        self.fc1 = nn.Linear(out_2*13*13, n_class) # => 2704

        

    # Prediction

    def forward(self, x, i=None):

        x = self.maxpool1(F.relu(self.conv1(x)))

        x = self.maxpool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        if i is not None and i < 1:

            print("x.shape:", x.shape)

#             print("weight, bias:",self.state_dict())

#             print("weight:",self.linear.weight)

#             print("bias:",self.linear.bias)

        yhat = F.softmax(self.fc1(x), dim=1)

#         yhat = F.softmax(self.fc1(x), dim=-1)

        return yhat



    # Outputs activation, this is not necessary

    def activations(self, x):

        z1 = self.conv1(x)

        a1 = F.relu(z1)

        out1 = self.maxpool1(a1)

        z2 = self.conv2(out1)

        a2 = F.relu(z2)

        out2 = self.maxpool1(a2)

        return z1,a1,z2,a2,out2,out1.view(out1.size(0),-1)

model = Net()

#print("The parameters: ", list(model.parameters()))

plot_parameters(model.state_dict()['conv1.weight'], number_rows=4, name="1st layer kernels before training ")

plot_parameters(model.state_dict()['conv2.weight'], number_rows=4, name='2nd layer kernels before training' )
n_epochs = 5

learning_rate = 0.1

momentum = 0.1

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

criterion = nn.CrossEntropyLoss()

gradient_plot = plot_diagram()

input_dim = 3*227*227

output_dim = 2 # Softmaxを使うため、バイナリではなく、2クラスとする



def train(model, train_loader, validate_loader, optimizer, criterion, dataset_val, epochs, print_flg=True):

    results = {'train_loss': [], 'validation_acc': []}

    for epoc in range(epochs):

        print(epoc)

        # train

        COST = 0

        for i, (x, y) in enumerate(train_loader):

#             x_reshaped = x.view(-1, input_dim)

            if print_flg is True:

#                 print(x.size(0), x.size(1))

                print("The shape of x: ", x.shape)

                print_flg = False

            optimizer.zero_grad()

            outputs = model(x, i)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            # plot the diagram for us to have a better idea

#             gradient_plot(loss.data.item(), epoc)

            COST += loss.data

        results['train_loss'].append(COST)

        print('cost:', COST)



        # evaluate

        correct = 0

        for x,y in validate_loader:

#             x_reshaped = x.view(-1, input_dim)

            outputs = model(x)

            _, predicted = torch.max(outputs, 1) # choose 1 class from 2 classes

            correct += (predicted == y).sum().item()

        accuracy = 100 * (correct / len(dataset_val))

        results['validation_acc'].append(accuracy)

        print('accuracy:', accuracy)

    return results

# Train Model with 5 epochs

training_results = train(model, train_loader, val_loader, optimizer, criterion, dataset_val, n_epochs)

# Plot the loss and accuracy

fig, ax1 = plt.subplots()

color = 'tab:red'

ax1.plot(training_results['train_loss'], color=color)

ax1.set_xlabel('epoch', color=color)

ax1.set_ylabel('Cost', color=color)

ax1.tick_params(axis='y', color=color)

    

ax2 = ax1.twinx()  

color = 'tab:blue'

ax2.set_ylabel('accuracy', color=color) 

ax2.set_xlabel('epoch', color=color)

ax2.plot(training_results['validation_acc'], color=color)

ax2.tick_params(axis='y', color=color)

fig.tight_layout()

print(training_results['train_loss'])
show_data(dataset_train[0])
# Determine the activations

out = model.activations(dataset_train[0][0].view(1, 3, 227, 227))

# /ZFNet-MLP-Mix/blob/master/samples/michigan/PyTorch/9.5_CNN_How_To_Plot_Kernel_Parameters.ipynb参照



# Plot the outputs after the first CNN

plot_activations(out[0], number_rows=4, name="Output after the 1st CNN")
# Plot the outputs after the first Relu

plot_activations(out[1], number_rows=4, name="Output after the 1st Relu")
# Plot the outputs after the second CNN

plot_activations(out[2], number_rows=4, name="Output after the 2nd CNN")
# Plot the outputs after the second Relu

plot_activations(out[3], number_rows=4, name="Output after the 2nd Relu")
# Plot the channels



plot_channels(model.state_dict()['conv1.weight'])

plot_channels(model.state_dict()['conv2.weight'])
import torchvision.models as models

model = models.resnet18(pretrained=True)



# Set no trainable to parameters

for param in model.parameters():

    param.requires_grad = False



# Replace the output layer with Liner layer(2 classes)

model.fc = nn.Linear(512, 2)

# print(model)

# print(model.fc.weight.requires_grad)

for parameter in model.parameters():

    print(parameter.requires_grad)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)
import time

n_epochs = 1

start_time = time.time()

def train(model, train_loader, validate_loader, optimizer, criterion, dataset_val, epochs):

    results = {'train_loss': [], 'validation_acc': []}

    for epoc in range(epochs):

        print(epoc)

        # train

        COST = 0

        for x, y in train_loader:

            model.train() # to do dropout

            optimizer.zero_grad() # clear gradient 

            outputs = model(x) # make a prediction 

            loss = criterion(outputs, y) # calculate loss 

            loss.backward() # calculate gradients of parameters 

            optimizer.step() # update parameters 

            COST += loss.data

            results['train_loss'].append(loss.data)

        print('cost:', COST)



        # evaluate

        correct = 0

        for x_test,y_test in validate_loader:

            model.eval() # not to do dropout

            outputs = model(x_test)

            _, predicted = torch.max(outputs, 1) # find max

            correct += (predicted == y_test).sum().item()

        accuracy = correct / len(dataset_val)

        results['validation_acc'].append(accuracy)

        print('accuracy:', accuracy)

    return results



training_results = train(model, train_loader, val_loader, optimizer, criterion, dataset_val, n_epochs)
plt.plot(training_results['train_loss'])

plt.xlabel("iteration")

plt.ylabel("loss")

plt.show()

# Plot the misclassified samples



count = 0

i = 0

for x_test, y_test in dataset_val:

#     outputs = model(x_test)

    outputs = model(x_test.view(-1, 3, 227, 227)) # batch形式に変換

    _, yhat = torch.max(outputs, 1)

    if yhat != y_test:

        show_data((x_test, y_test))

        plt.show()

#         print("yhat:", yhat)

#         print("actual ", y_test)

        print('sample {0} predicted value: {1} actual value:{2}'.format(i, yhat, y_test))

        count += 1

    i += 1

    if count >= 4:

        break