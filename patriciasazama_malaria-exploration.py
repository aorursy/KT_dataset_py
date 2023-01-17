# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

clean_cell_files = []

infected_cell_files = []



for dirname, _, filenames in os.walk('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected'):

    for filename in filenames:

        clean_cell_files.append(os.path.join(dirname, filename))

        

for dirname, _, filenames in os.walk('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized'):

    for filename in filenames:

        infected_cell_files.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline
# Let's explore our dataset



# Get number of samples for each class (clean/infected)

print("clean sample count:", len(clean_cell_files))

print("infected sample count:", len(infected_cell_files))



sample = imageio.imread(clean_cell_files[0])

print("sample dimensions: ", sample.shape)

print("min and max values: ", np.min(sample), np.max(sample))

temp_shape = sample.shape



# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    sample = imageio.imread(clean_cell_files[indx])

    plt.imshow(sample)

    

# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    sample = imageio.imread(infected_cell_files[indx])

    plt.imshow(sample)

    

# From this we have a rough idea what the images look like, and we know that the samples are nonunifrom sizes.
dims = (200, 200, 3)

dims = (50, 50, 3)
from skimage.transform import resize



clean_cells = []

clean_cells_dims = []

for indx in range(len(clean_cell_files)):

    try:

        sample = imageio.imread(clean_cell_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        clean_cells_dims.append(sample)

        clean_cells.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", clean_cell_files[indx])

print("clean cell count:", len(clean_cells))

    

infected_cells = []

infected_cells_dims = []

for indx in range(len(infected_cell_files)):

    try:

        sample = imageio.imread(infected_cell_files[indx])

        sample = np.array(sample)

        sample = resize(sample, dims) # reshape so all images are same dimensions

        infected_cells_dims.append(sample)

        infected_cells.append(sample.flatten()) # flatten each sample so it's the correct shape for umap clustering

    except:

        print("> error loading image: ", infected_cell_files[indx])

print("infected cell count:", len(infected_cells))
# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    #plt.imshow(np.transpose(data_dims[indx], (1,2,0)))

    plt.imshow(infected_cells_dims[indx])
!pip install umap-learn==0.4
# Get the data in a more traditional ML processing format (data and label arrays)

n = 10000 # change to subsample if loading full dataset is too large

n = None



print(len(clean_cells), len(infected_cells))



if n is not None:

    clean_labels = np.zeros(len(clean_cells[:n]))

    infected_labels = np.ones(len(infected_cells[:n]))

    target = np.concatenate([clean_labels[:n], infected_labels[:n]])



    data = clean_cells[:n] + infected_cells[:n]

    data_dims = clean_cells_dims[:n] + infected_cells_dims[:n]

else:

    clean_labels = np.zeros(len(clean_cells))

    infected_labels = np.ones(len(infected_cells))

    target = np.concatenate([clean_labels, infected_labels])



    data = clean_cells + infected_cells

    data_dims = clean_cells_dims + infected_cells_dims

data = np.array(data)

data_dims = np.array(data_dims)



from sklearn.utils import shuffle

data, data_dims, target = shuffle(data, data_dims, target, random_state=0)



print(data.shape, data_dims.shape, target.shape)
# Let's try some dimensionality reduction based clustering: UMAP



import umap

print(umap.__version__)



reducer = umap.UMAP(low_memory=True)

embedding = reducer.fit_transform(data)



fig, ax = plt.subplots(figsize=(12, 10))

color = target

plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=2)

plt.setp(ax, xticks=[], yticks=[])

plt.title("Malaria data embedded into two dimensions by UMAP", fontsize=18)



plt.show()
malaria = pd.DataFrame({"x":embedding[:, 0], "y":embedding[:, 1], "color":target})#, "image":data})
import plotly.express as px



fig = px.scatter(malaria, x="x", y="y", color="color", title="Malaria data embedded into two dimensions by UMAP")



fig.show()
# View the first few samples

for indx in range(5):

    plt.figure() # don't overwrite other plots

    #plt.imshow(np.transpose(data_dims[indx], (1,2,0)))

    plt.imshow(data_dims[indx])
print(data_dims.shape)

data_dims = np.transpose(data_dims, (0,3,1,2))

print(data_dims.shape)



print(target.shape)

target = np.expand_dims(target, axis=-1)

print(target.shape)



import math



train_data = data_dims[:math.floor(len(data_dims)*.95)]

train_target = target[:math.floor(len(target)*.95)]



test_data = data_dims[math.floor(len(data_dims)*.95):]

test_target = target[math.floor(len(target)*.95):]
# Build a CNN to classify infected/clean cell images (dims)



import torch

import torch.nn as nn

import torch.nn.functional as F



inner1 = 70

inner2 = 10



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(dims[2], inner1, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(inner1, inner2, 5)

        self.dense = nn.Linear(inner2*9*9, 1)



    def forward(self, x):

        #print(x.shape)

        x = self.pool(F.relu(self.conv1(x)))

        #print(x.shape)

        x = self.pool(F.relu(self.conv2(x)))

        #print(x.shape)

        x = x.view(-1, inner2*x.shape[2]*x.shape[2])

        #print(x.shape)

        return torch.sigmoid(self.dense(x))
model = Model()
# data, target

out = model(torch.from_numpy(data_dims[:5]).float())

print(out.shape)
import torch.optim as optim



criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
with torch.no_grad():

    model.eval()

    

    yhat = model(torch.from_numpy(test_data).float())

    test_loss = criterion(torch.from_numpy(test_target).float(), yhat)

    print("testing loss:", test_loss)

    

    #print([(test_target[i][0] == round(float(yhat[i][0]))) for i in range(len(test_target))])

    

    acc = [test_target[i][0] == round(float(yhat[i][0])) for i in range(len(test_target))]

    print("testing accuracy: ", sum(acc), "/", len(test_target), "=", sum(acc)*1.0/len(test_target))
batch_size = 50

batch_num = math.floor(len(train_data)/batch_size)

print_batch = 100



model.train()



for epoch in range(50):  # loop over the dataset multiple times



    running_loss = 0.0

    for i in range(batch_num):

        inputs = train_data[i*batch_size:(i+1)*batch_size]

        labels = train_target[i*batch_size:(i+1)*batch_size]



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = model(torch.from_numpy(inputs).float())

        loss = criterion(outputs, torch.from_numpy(labels).float())

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if (i+1) % print_batch == 0:    # print every 100 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i+1, running_loss / print_batch))

            running_loss = 0.0



print('Finished Training')
with torch.no_grad():

    model.eval()

    

    yhat = model(torch.from_numpy(test_data).float())

    test_loss = criterion(torch.from_numpy(test_target).float(), yhat)

    print("testing loss:", test_loss)

    

    #print([(test_target[i][0] == round(float(yhat[i][0]))) for i in range(len(test_target))])

    

    acc = [test_target[i][0] == round(float(yhat[i][0])) for i in range(len(test_target))]

    print("testing accuracy: ", sum(acc), "/", len(test_target), "=", sum(acc)*1.0/len(test_target))