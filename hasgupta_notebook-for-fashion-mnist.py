# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from fastai.imports import *

from fastai.torch_core import *

from fastai.core import *

from fastai.metrics import *

from fastai.datasets import *

from torch.autograd import Variable
df_train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv",low_memory = False)
df_test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv",low_memory = 0)
df_train.head()
df_test.head(2)
x_train = df_train.drop('label',axis= 1)

x_train.head()
x_test = df_test.drop('label',axis = 1)

x_test.head()
y_train = df_train['label']
y_test = df_test['label']
x_train = (np.matrix(x_train))
x_test = (np.matrix(x_test))
y_train = (np.array(y_train))
y_test = (np.array(y_test))
print(type(y_train))

print(y_train[:10])

print(type(x_train))

print(x_train[:5])

print(type(x_test))

print(x_test.shape)

print(x_test[:5])
def show(im,title = None):

    values = {0:'Tshirt/Top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle Boot'}

    plt.imshow(im,cmap='inferno')

    if title is not None:

        plt.title(values[int(title)],fontsize = 15)
def plots(imgs,fsize=(12,6),rows = 2, titles = None):

    values = {0:'Tshirt/Top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle Boot'}

    fig = plt.figure(figsize =(12,6))

    columns = int(len(imgs)//rows)

    for i in range(len(imgs)):

        sp = fig.add_subplot(rows,columns,i+1)

        sp.axis('Off')

        if titles is not None:

            sp.set_title(values[int(titles[i])],fontsize = 19)

        sp.imshow(imgs[i],cmap = "Greys")
images = np.array(x_train).reshape(-1,28,28)
images_test = np.array(x_test).reshape(-1,28,28)
image_tensor_test = torch.from_numpy(images_test).cuda()
image_tensor = torch.from_numpy(images).cuda()
show((image_tensor[123].cpu()),(y_train[123]))
plots(images[:12],titles = y_train[:12])
mean = x_train.mean()

std = x_train.std()

print("Mean is :",mean," Standard Deviation is :",std)
print("Initial mean and std for training and test...\n")

print("Training\n Mean :",x_train.mean(),"\t Std :",x_train.std())

print("Testing\n Mean :",x_test.mean(),"\t Std :",x_test.std())

x_train = (x_train-mean)/std

x_test = (x_test-mean)/std

print("After normalization\n")

print("Training\n Mean :",x_train.mean(),"\t Std :",x_train.std())

print("Testing\n Mean :",x_test.mean(),"\t Std :",x_test.std())
show(images_test[10],y_test[10])
net = nn.Sequential(

    nn.Linear(784, 100),

    nn.ReLU(),

    nn.Linear(100,100),

    nn.ReLU(),

    nn.Linear(100, 100),

    nn.ReLU(),

    nn.Linear(100,10),

    nn.LogSoftmax()

    ).cuda()

# Starting simple with a Logistic regression

learn_rate = 1e-4

loss = nn.NLLLoss() 

# Negative Log Likelihood Loss

opt = optim.Adam(net.parameters(),lr = learn_rate)

# STOCHASTIC GRADIENT DESCENT
x_train = torch.from_numpy(x_train).cuda()

x_test = torch.from_numpy(x_test).cuda()

y_train = torch.from_numpy(np.array(y_train)).cuda()

y_test = torch.from_numpy(np.array(y_test)).cuda()
x_train = x_train.float()

x_test = x_test.float()

class Dataset(torch.utils.data.Dataset):

    'Characterizes a dataset for pytorch'

    def __init__(self,train_data,train_labels):

        # Initialization of 

        # training data and training labels

        self.labels = train_labels

        self.data = train_data

        

        

    def __len__(self):

        

        'Denotes the total number of samples'

        return (len(self.data))

    

    def __getitem__(self,index):

        'Generates a single sample of data from training'

        # Selecting Sample

        # the sample index

        ID = index

        #Loading sample and the label of data.

        X = x_train[ID]

        Y = y_train[ID]

        

        return X,Y
params = {'batch_size':128,

         'shuffle':True,

         'pin_memory':False}

max_epochs = 10

 

use_cuda = torch.cuda.is_available()

#device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True

# This is just loading the data and passing on to the generator.

train_set = Dataset(x_train,y_train)

train_gen = torch.utils.data.DataLoader(train_set,**params)

accuracies = []

losses = []

for epoch in range(15):

    count = 0

    for batch,labels in train_gen:

        y_pred = net(Variable(batch).cuda())

        l = loss(y_pred,Variable(labels).cuda())

            # Variable instance tells the pytorch mocdule to keep track of 

            # the derivatives of each parameter as you go on...



            #acu = np.mean(to_np(y_pred).argmax(axis=1)==to_np(labels))

            #print("Accuracy is ",acu," Loss is ",l)

    

        #Before calculation of gradients, all the gradients in the current step are set to zero.

        opt.zero_grad()

        # This l.backward is used to compute all the gradients wrt the parameters of the neural net.

        l.backward()

        # This is the step when the optimizer uses gradient descent by calculating partial derivatives wrt each of the

        # parameters and goes in that direction(updates value of parameter) where loss decreases by a step learning_rate*derivative value

        opt.step()

    y_preds = net(Variable(x_test[:]).cuda())

    #print("Accuracy for all test images")

    ac = np.mean(to_np(y_preds).argmax(axis=1) == to_np(y_test[:]))

    accuracies.append(ac)

    losses.append(int(loss(y_preds,Variable(y_test[:]).cuda())))

    plt.scatter(epoch,ac)

    plt.scatter(epoch,int(loss(y_preds,Variable(y_test[:]).cuda())))

plt.grid()

plt.xlabel("Number of Epochs")

plt.plot([i for i in range(15)],accuracies,color = 'black',label = 'Accuracy')

plt.plot([i for i in range(15)],losses,color = 'cyan',label = 'Loss')

plt.legend(frameon = 1, fontsize = 14)

plt.title("Accuracy and Loss plot",fontsize = 17)

plt.savefig("Losses And Accuracy in Fashion MNIST.png",dpi =200)



y_preds = net(Variable(x_test[:]).cuda())

print("Accuracy for all test images")

print(np.mean(to_np(y_preds).argmax(axis=1) == to_np(y_test[:])))
y_pred = net(Variable(x_test[900:912]).cuda())
plots(images_test[900:912],titles=to_np(y_pred.cpu().argmax(axis=1)))
plots(images_test[900:912],titles=to_np(y_test[900:912] ))