# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import numpy as np

from tqdm import tqdm

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import shutil

import zipfile

if(os.path.isdir('/kaggle/working/train/')):

    shutil.rmtree('/kaggle/working/train/')

    shutil.rmtree('/kaggle/working/test1/')

    os.remove('training_data.npy')

    os.remove('model.log')

else:

    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:

        z.extractall(".")

    os.makedirs('/kaggle/working/train/cat')

    os.makedirs('/kaggle/working/train/dog')

    for dir in os.listdir('/kaggle/working/train'):

        dataDir = '/kaggle/working/train/'+dir

        if("cat" in dataDir):

            shutil.move(dataDir, "/kaggle/working/train/cat")

        else:

            shutil.move(dataDir, "/kaggle/working/train/dog")    

    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:

        z.extractall(".")
REBUILD_DATA = True



if torch.cuda.is_available():

    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 

    print("Running on the GPU")

else:

    device = torch.device("cpu")

    print("Running on the CPU")

    

class DogsVSCats():

    IMG_SIZE = 50

    CATS = "/kaggle/working/train/cat"

    DOGS = "/kaggle/working/train/dog"

    TESTING = "/kaggle/working/train/test1"

    LABELS = {CATS: 0, DOGS: 1}

    training_data = []

    catcount = 0

    dogcount = 0



    def make_training_data(self):

        for label in self.LABELS:

            print(label)

            for f in tqdm(os.listdir(label)):

                if "jpg" in f:

                    try:

                        path = os.path.join(label, f)

                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 

                        #print(np.eye(2)[self.LABELS[label]])



                        if label == self.CATS:

                            self.catcount += 1

                        elif label == self.DOGS:

                            self.dogcount += 1



                    except Exception as e:

                        pass

                        #print(label, f, str(e))



        np.random.shuffle(self.training_data)

        np.save("training_data.npy", self.training_data)

        print('Cats:',dogsvcats.catcount)

        print('Dogs:',dogsvcats.dogcount)



class Net(nn.Module):

    def __init__(self):

        super().__init__() # just run the init of parent class (nn.Module)

        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window

        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window

        self.conv3 = nn.Conv2d(64, 128, 5)



        x = torch.randn(50,50).view(-1,1,50,50)

        self._to_linear = None

        self.convs(x)



        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.

        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).



    def convs(self, x):

        # max pooling over 2x2

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))



        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x



    def forward(self, x):

        x = self.convs(x)

        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 

        x = F.relu(self.fc1(x))

        x = self.fc2(x) # bc this is our output layer. No activation here.

        return F.softmax(x, dim=1)

        

        

if REBUILD_DATA:

    dogsvcats = DogsVSCats()

    dogsvcats.make_training_data()



net = Net().to(device)

print(net)



training_data = np.load("training_data.npy", allow_pickle=True)

print(len(training_data))

    

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)

X = X/255.0

y = torch.Tensor([i[1] for i in training_data])

    

VAL_PCT = 0.1  

val_size = int(len(X)*VAL_PCT)



train_X = X[:-val_size]

train_y = y[:-val_size]



test_X = X[-val_size:]

test_y = y[-val_size:] 
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.MSELoss()

def fwd_pass(X, y, train=False):

    if train:

        net.zero_grad()

    outputs = net(X)

    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]

    acc = matches.count(True)/len(matches)

    loss = loss_function(outputs, y)



    if train:

        loss.backward()

        optimizer.step()



    return acc, loss



def test(size=32):

    random_start = np.random.randint(len(test_X) - size)

    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]

    with torch.no_grad():

        val_acc, val_loss = fwd_pass(X.view(-1,1,50,50).to(device), y.to(device))

    return val_acc, val_loss



val_acc, val_loss = test(size=1000)

print(val_acc, val_loss)
import time



MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time. 

net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.MSELoss()



def train(net):

    BATCH_SIZE = 100

    EPOCHS = 8



    with open("model.log", "a") as f:

        for epoch in range(EPOCHS):

            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):

                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)

                batch_y = train_y[i:i+BATCH_SIZE]



                batch_X, batch_y = batch_X.to(device), batch_y.to(device)



                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                

                if i % 10 == 0:

                    val_acc, val_loss = test(size=100)

                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

train(net)