import torch

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



torch.__version__
#helper function to summarize the properties of a tensor

def describe(x):

    print("Type :: {}".format(x.type()))

    print("Data Type :: {}".format(x.dtype))

    print("Size :: {}".format(x.size()))

    print("Shape :: {}".format(x.shape))

    print("Number of elements :: {}".format(x.numel()))

    # Dimension :: x.ndimension()

    print("Dimension :: {}".format(x.dim()))

    print("Data :: \n{}".format(x))

    print('-----'*5)
#Scalar (o-d Tensor)

s = torch.tensor(2504)

describe(s)
m = torch.Tensor([25, 4])

describe(m)
m = m.type(torch.LongTensor)

describe(m)
#Using torch.tensor(), can specify data types

m = torch.tensor([25, 4], dtype=torch.int32)

describe(m)
int_list = [1,2,3,4,5]

float_list = [1.0,2.0,3.0,4.0,5.0]



int_tensor = torch.tensor(int_list)

describe(int_tensor)



float_int_tensor = torch.tensor(float_list, dtype=torch.int64)

describe(float_int_tensor)



float_tensor = torch.tensor(float_list)

describe(float_tensor)



int_float_tensor = torch.FloatTensor(int_list)

describe(int_float_tensor)



new_flot_tensor = torch.tensor(int_list, dtype=torch.float64)

describe(new_flot_tensor)
new_flot_tensor.tolist()
#Constructing a Tensor directly 

#Creates an unitilised matrix of size 5x4

c = torch.Tensor(5,4)

describe(c)
#Initialized with numpy array

n = torch.tensor(np.array([25, 4], dtype=np.int32))

describe(n)
a = np.random.rand(10)

tensor_a = torch.from_numpy(a)

describe(tensor_a)
back_to_numpy_a = tensor_a.numpy()

back_to_numpy_a, back_to_numpy_a.dtype
pd_series = pd.Series(np.arange(1,11,2))

print(pd_series)

tensor_fron_series = torch.from_numpy(pd_series.values)

describe(tensor_fron_series)
df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})

print(df)

tensor_fron_dataframe = torch.from_numpy(df.values)

describe(tensor_fron_dataframe)
empt = torch.empty(10)

describe(empt)
z = torch.zeros(2,3,4)

describe(z)
#torch.ones()

#Constructing a tensor using the existing Tensor

o = torch.ones_like(z)

describe(o)
#can also create tensor filled with same value

#torch.fill(shape)

z.fill_(25) #_ ===> in-place operation

describe(z)
#Creating a diagonal matrix tensor using the data

d = torch.diag(torch.Tensor([1,2,3,4]))

describe(d)
#Creating an identity matrix

i = torch.eye(5,5)

describe(i)
#Creates a tensor insitialised with 10 uniform random values

x = torch.rand(10)

describe(x)
#Creating a normal distribution tensor of shape x

x_normal = torch.randn_like(x)

describe(x_normal)
# randint(start, end, size(must be a tuple))

rand_ints = torch.randint(0, 100, (5, 4))

describe(rand_ints)
#linspace(start, end, number of elements)

ls = torch.linspace(20, 30, 25)

describe(ls)
#range(start, end, skip)

rg = torch.range(0, 100, 3)

describe(rg)
rand_ints.data
#Indexing and Slicing

#3rd row, 2nd and 3rd column

rand_ints[2, 1:3]
rand_ints[2, 1:3] = torch.Tensor([19, 91])

rand_ints.data
#first column

rand_ints[:, 0]
#first 2 row

rand_ints[:2,:]
#last 3 column

rand_ints[:,-3:]
#last row

rand_ints[-1,:]
#Access 2nd and 4th col

indices = torch.LongTensor([1,3]) # Index must be integers

describe(torch.index_select(rand_ints, dim=1, index=indices))
#access 2nd and 4th row

describe(torch.index_select(rand_ints, dim=0, index=indices))
a = torch.arange(1, 21, dtype=torch.int32)

describe(a)

reshaped_a = a.view(2,2,5)

describe(reshaped_a)
# For dynamic size arrays or when size is unknown

# -1 is inferred from other dimension

reshaped_a = a.view(-1,1)

describe(reshaped_a)

reshaped_a = a.view(10,-1)

describe(reshaped_a)
# For 3d tensors, dim=0 represents 2D tenasors, dim=1 represents rows, dim=2 represents column

a = a.view(2,2,5)

a.sum(), a.sum(dim=0), a.sum(dim=1), a.sum(dim=2)
a = a.view(4, 5)

describe(torch.cat((a,a), dim=1))

print("\n")

describe(torch.cat((a,a), dim=0))
tnsr = torch.linspace(0, 2*np.pi, 100)

sin_tnsr = torch.sin(tnsr)
%matplotlib inline
plt.plot(tnsr.numpy(), sin_tnsr.numpy())
#matrix of size 506x13

from sklearn.datasets import load_boston

boston = load_boston()



boston.DESCR
boston.feature_names
boston.target
# from_numpy converts numpy array to pytorch tensor

boston_data = torch.from_numpy(boston.data)

boston_data.size()
#slicing element from columns 3 to 7 from first 2 rows of a tensor

boston_data[:2, 3:8]
#Arithmetic operations +-*/

a = torch.rand(2,2)

b = torch.rand(2,2)



c = a+b

c
#_ signifies for inplace operation(Here, addition)

a.add_(b)
one_value_tensor = torch.tensor([25], dtype=torch.int32)

describe(one_value_tensor)
one_value_tensor.item(), c[0][0], c[0][0].item()
#Linear Algebra

x1 = torch.arange(6).view(2,3)

x2 = torch.randint(1, 11, (3,1))

print("x1", "\n============")

describe(x1)

print("============\n")

print("x2", "\n============")

describe(x2)

print("============\n")

print("x1 matmul x2", "\n============")

describe(torch.matmul(x1, x2))

print("============\n")

print("x1 transpose", "\n============")

describe(torch.transpose(x1, 0, 1))

print("============\n")
#Vector Dot Product

v1 = torch.tensor([1,2,3])

v2 = torch.tensor([4,5,6])

v1.dot(v2)
#Hadamard Product (Element wise product)

v1*v2
t = torch.ones(2,5)

print(t.requires_grad)
#x is a tensor initialised with value 2

#requires_grad as True, it starts to track all operations on it.

x = torch.tensor(2, dtype=torch.float64, requires_grad=True)

print(x)

#y = f(x)

y = x**2

print(y)
print("x ::", x)

print("x data :: ", x.data)

print("x  gradient :: ", x.grad)

print("x gradient function :: ", x.grad_fn)
print("y ::", y)

print("y data :: ", y.data)

print("y  gradient :: ", y.grad)

print("y gradient function :: ", y.grad_fn)
#finish all computation then can call .backward() and have all the gradients computed automatically.

y.backward()
print("x ::", x)

print("x data :: ", x.data)

print("x  gradient :: ", x.grad)

print("x gradient function :: ", x.grad_fn)
print("y ::", y)

print("y data :: ", y.data)

print("y  gradient :: ", y.grad)

print("y gradient function :: ", y.grad_fn)
x = torch.tensor(2, dtype=torch.float64, requires_grad=True)

y = x ** 2

z = (y ** 3) * 3



z.backward()
print("x ::", x)

print("x data :: ", x.data)

print("x  gradient :: ", x.grad)

print("x gradient function :: ", x.grad_fn)
print("y ::", y)

print("y data :: ", y.data)

print("y  gradient :: ", y.grad)

print("y gradient function :: ", y.grad_fn)
print("z ::", z)

print("z data :: ", z.data)

print("z  gradient :: ", z.grad)

print("z gradient function :: ", z.grad_fn)
u = torch.tensor(2, dtype=torch.float64)

u.requires_grad_(True)

print(u)
v = torch.tensor(3, dtype=torch.float64, requires_grad=True)

print(v)
# y = f(u, v)

y = u*v + u**2

print(y)
y.backward()
print("u ::", u)

print("u data :: ", u.data)

print("u  gradient :: ", u.grad)

print("u gradient function :: ", u.grad_fn)
print("v ::", v)

print("v data :: ", v.data)

print("v  gradient :: ", v.grad)

print("v gradient function :: ", v.grad_fn)
print("y ::", y)

print("y data :: ", y.data)

print("y  gradient :: ", y.grad)

print("y gradient function :: ", y.grad_fn)
print(t)
t.requires_grad_(True)

print(t)

with torch.no_grad():

    print((t**2).requires_grad)
#To stop a tensor from tracking history, you can call .detach()

#particularly helpful when evaluating a model 

#because the model may have trainable parameters with requires_grad=True, 

#but for which we don’t need the gradients.

print(t)

t.detach_()

print(t)
x = torch.linspace(-10, 10, 10, requires_grad = True)

w = torch.linspace(-10, 10, 10, requires_grad = True)



Y = x ** 2

Z = torch.relu(w)



y = torch.sum(Y)

z = torch.mean(Z)



#torch.autograd cannot compute full Jacobian directly

#input to backward must be scalar.

#if it is not scalar, need to pass a vector for vector jacobian product.

y.backward()

z.backward()



plt.figure(figsize=(12, 5))



ax0 = plt.subplot(121)

ax0.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')

ax0.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')

ax0.set_xlabel('x')

ax0.set_title('Square Function')



ax1 = plt.subplot(122)

ax1.plot(w.detach().numpy(), Z.detach().numpy(), label = 'function')

ax1.plot(w.detach().numpy(), w.grad.detach().numpy(), label = 'derivative')

ax1.set_xlabel('w')

ax1.set_title('Relu Function')



plt.legend()

plt.show()
# Variables are wrapper around the tensor ith gradient and reference to a function that created it.

from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)

x
#Tensors can be moved to any device.

#Following code checks if GPU is available, maked cuda (GPU) default device.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x3 = torch.rand(2,5).to(device)

if device == "cuda":

    print(torch.cuda.get_device_name(0))

    print(x3.type())

else:

    print(device)

    print(x3.type())
from torch.utils.data import Dataset

#Forces to give same random number every time it gets compiled

torch.manual_seed(1)
#Class to load sample dummy dataset

class toyDataset(Dataset):

    #Constructor with default values

    def __init__(self, length=10, transform=None):

        self.len = length

        #x :: input features

        self.x = 2 * torch.randint(0, 101,(length, 2), dtype=torch.float64)

        #y :: target labels

        self.y = torch.ones(length, 1)

        #Whether data need to transformed (like, normalization, etc)

        self.transform = transform

        

    #Method overriding to return the total number of instances 

    def __len__(self):

        return self.len

    

    #Method overriding to return data samples

    def __getitem__(self, index):

        sample = self.x[index], self.y[index]

        if self.transform:

            sample = self.transform(sample)

        return sample
#Creating instance of toyDataset and accessing example instances

data = toyDataset()

for i in range(5):

    print(data[i])
class transform_my_data(object):

    def __init__(self, tranformation_params):

        """

            Constructor

        """

        self.tranformation_params = tranformation_params

    

    def __call__(self, sample):

        """

            Executor:

            Necessary tranformation

            to each instance of data.

            

        """

        x, y = sample

        x *= self.tranformation_params

        

        return x, y       
class normalise_my_data(object):

    def __init__(self, total_instances):

        """

            Constructor

        """

        self.total_instances = total_instances

    

    def __call__(self, sample):

        """

            Executor:

            Necessary tranformation

            to each instance of data.

        """

        x, y = sample

        x /= self.total_instances

        

        return x, y
print(len(data))
transform = transform_my_data(0.2)

normalise = normalise_my_data(len(data))
transformed_dataset = toyDataset(transform=transform)
normalised_dataset = toyDataset(transform=normalise)
# Use loop to print out first 10 elements in dataset



for i in range(5):

    print(data[i])

    print(transformed_dataset[i])

    print(normalised_dataset[i])
from torchvision import transforms
data_transform = transforms.Compose([transform, normalise])

data_transform
data_transform(data[0])
dataset = toyDataset(transform=data_transform)
for i in range(5):

    print(dataset[i])
from matplotlib.pyplot import imshow

import matplotlib.pylab as plt

from PIL import Image

import os
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
directory = "/kaggle/input/100-sample-fmnist/"

csv_file = "index.csv"

csv_path = os.path.join(directory+csv_file)
data_name = pd.read_csv(csv_path)

data_name.head()
#Filename, Label/class

data_name.iloc[0,1], data_name.iloc[0,0]
image_name = data_name.iloc[0,1]

image_path = os.path.join(directory+"img/"+image_name)

image = Image.open(image_path)

plt.imshow(image,cmap='gray', vmin=0, vmax=255)

plt.title(data_name.iloc[0, 0])

plt.show()
class fashionDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.root_dir = root_dir

        csv_path = os.path.join(root_dir+csv_file)

        self.csv_file = pd.read_csv(csv_path)

        self.transform = transform

    

    def __len__(self):

        return len(self.csv_file)

    

    def __getitem__(self, idx):

        """

            to fetch instances of dataset

            idx :: index

        """

        #Loading the image

        img_name = os.path.join(self.root_dir+"img/"+self.csv_file.iloc[idx, 1])

        image = Image.open(img_name)

        

        #Loading the label

        label = self.csv_file.iloc[idx, 0]

        

        #Applying transformation

        if self.transform:

            image = self.transform(image)

            

        return image, label
#Creating object of dataset

fdata = fashionDataset(csv_file, directory)
#fetching length of total instances

len(fdata)
#Accessing a particulare instance of a dataset

img = fdata[100]



plt.imshow(img[0],cmap='gray', vmin=0, vmax=255)

plt.title(img[1])

plt.show()
img_transformation = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])

newFData = fashionDataset(csv_file=csv_file, root_dir=directory, transform=img_transformation)
#function to load the image for display

def show(sample, shape=(28,28)):

    plt.imshow(sample[0].numpy().reshape(shape), cmap='gray')

    plt.title(sample[1])

    plt.show()
#Accessing a particulare instance of a dataset

img = newFData[100]

show(img, shape=(20,20))
for i in range(5):

    show(newFData[i], shape=(20,20))
from torch.utils.data import DataLoader
dataloader = DataLoader(newFData, batch_size=5, shuffle=True, num_workers=5)
for i_batch, sample_batched in enumerate(dataloader):

    for i in range(sample_batched[0].shape[0]):

        sample = (sample_batched[0][i], sample_batched[1][i])

        show(sample, shape=(20,20))

    if i_batch == 2:

        break
import torchvision.datasets as dsets
#importing the pre-built mnist dataset

mnist_dataset = dsets.MNIST(root='./data',

                           train=False, # If True, creates dataset from training.pt, otherwise from test.pt.

                           download=True,

                           transform = transforms.ToTensor())
mnist_dataset[0][0].shape
mnist_dataset[0][1]
show(mnist_dataset[0])
#Using the previously referred boston housing dataset from sklearn

boston.feature_names
def get_input_data():

    inp_data = boston.data

    age = inp_data[:, 6]

    targets = boston.target

    

    X = Variable(torch.from_numpy(age).type(torch.FloatTensor), requires_grad=False).view(506,1)

    y = Variable(torch.from_numpy(targets).type(torch.FloatTensor), requires_grad=False)

    

    return X, y
#Defining Weights and bias for Linear Regression

W = Variable(torch.randn(1), requires_grad=True)

b = Variable(torch.randn(1), requires_grad=True)
lr = 1e-4

losses = []
X, y = get_input_data()

for i in range(10):

    y_pred = torch.matmul(X, W) + b

    

    loss = (y - y_pred).pow(2).mean()

    for param in [W, b]:

        if param.grad is not None:

            param.grad.data.zero_()

    loss.backward()

    loss = loss.data.item()

    losses.append(loss)

    if i%2 == 0:

        print(loss)

    

    W.data -= lr * W.grad.data

    b.data -= lr * b.grad.data
xplot = np.arange(10)

yplot = np.array(losses)

plt.plot(xplot,yplot)