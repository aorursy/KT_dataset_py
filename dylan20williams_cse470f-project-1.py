
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pal = sns.color_palette()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from torch.utils.data.dataset import Dataset

print('# File sizes')
for f in os.listdir('../input'):
    if not os.path.isdir('../input/' + f):
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
    else:
        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]
        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import *
from torch import from_numpy
import torch
IMG_PATH = '../input/planet-understanding-the-amazon-from-space/train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = '../input/planet-understanding-the-amazon-from-space/train_v2.csv'
TEST_DATA = '../'
TEST_PATH = '../input/planet-understanding-the-amazon-from-space/test-jpg-additional/'
TEST_DATA = '../input/planet-understanding-the-amazon-from-space/sample_submission_v2.csv'

class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#got this from https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab. Helps give me validation data because this dataset comes with none
class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''
    if random_seed!=None:
        np.random.seed(random_seed)

    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)

    return train, valid
transformations = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
transformation_augmented = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor(),normalize])
transformation_raw = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),normalize])

dset_whole = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
training_data, valid_data = train_valid_split(dset_whole)
im, target = valid_data[0]
target
def plot_sample(im, target):
    plt.imshow(im.numpy().transpose(1,2,0))
    #TODO show multple names
    plt.text(0,0, "Forest\nRiver", verticalalignment='top', color='yellow')
    
plot_sample(im, target)
import torch
df_train = pd.read_csv('../input/planet-understanding-the-amazon-from-space/train_v2.csv')
df_train.head()
labels = df_train['tags'].apply(lambda x: x.split(' '))
from collections import Counter, defaultdict
counts = defaultdict(int)
for l in labels:
    for l2 in l:
        counts[l2] += 1

data=[go.Bar(x=list(counts.keys()), y=list(counts.values()))]
layout=dict(height=800, width=800, title='Distribution of training labels')
fig=dict(data=data, layout=layout)
py.iplot(data, filename='train-label-dist')
com = np.zeros([len(counts)]*2)
for i, l in enumerate(list(counts.keys())):
    for i2, l2 in enumerate(list(counts.keys())):
        c = 0
        cy = 0
        for row in labels.values:
            if l in row:
                c += 1
                if l2 in row: cy += 1
        com[i, i2] = cy / c

data=[go.Heatmap(z=com, x=list(counts.keys()), y=list(counts.keys()))]
layout=go.Layout(height=800, width=800, title='Co-occurence matrix of training labels')
fig=dict(data=data, layout=layout)
py.iplot(data, filename='train-com')
import skimage.io

new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))
i = 0
for f, l in df_train[:9].values:
    img = skimage.io.imread('../input/planet-understanding-the-amazon-from-space/train-jpg/{}.jpg'.format(f))
    ax[i // 3, i % 3].imshow(img)
    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    i += 1
    
plt.show()

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from IPython.core.debugger import set_trace
BATCH_SIZE = 256
LEARNING_RATE = 0.01
WORKERS = 4
train_loader = DataLoader(training_data, batch_size = BATCH_SIZE, num_workers=WORKERS, shuffle = True)
test_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, num_workers=WORKERS, shuffle = True)
import torch.cuda
torch.cuda.is_available()
if torch.cuda.is_available():
    def togpu(x):
        return x.cuda()
    def tocpu(x):
        return x.cpu()
else:
    def togpu(x):
        return x
    def tocpu(x):
        return x
class SimpleCNN(nn.Module):
    def __init__(self, shape=(256,256), num_classes=17):
        super().__init__()
        
        #num_inputs = np.product(shape)
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3) 
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.layer3 = nn.Linear(2304, 256)
        self.layer4 = nn.Linear(256, 17)
        
    
    def forward(self, x):        
        
        #set_trace()
        x = self.layer1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
       
        
        x = self.layer2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        
        # Flatten
        x = x.view(x.size(0),-1)
        
        x = self.layer3(x)
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training)
        
        x = self.layer4(x)
        
        y = torch.sigmoid(x)
        
        return y  # Will learn to treat 'a' as the natural parameters of a multinomial distr. 
net = SimpleCNN(shape=(256,256), num_classes= 17)
net = togpu(net)
optimizer = torch.optim.SGD(params = net.parameters(), lr = LEARNING_RATE, momentum = 0.5)
start_epoch = 0
num_epochs = 30
best_eval_loss = float('inf')
import time
import tqdm
import sys
import shutil
#model = torch.load('../input/cse470f-project-1/simplecnn-checkpoint.pth.tar')
#model
def compute_eval_loss(net, loader):
    # Evaluate the model
    with torch.no_grad():
        eval_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(loader),
                                 file = sys.stdout,
                                 desc='Evaluating',
                                 total=len(loader),
                                 leave=False):
            inputs, labels = data
            inputs, labels = togpu(inputs), togpu(labels)
            outputs = net(inputs)               # Predict
            loss =  F.binary_cross_entropy(outputs, labels)   # Grade / Evaluate
            eval_loss += loss.item()
    eval_loss /= len(test_loader)
    return eval_loss
for epoch in tqdm.tnrange(start_epoch, num_epochs):
    
    running_loss = 0.0
    tstart = time.time()
    
    # Update the model parameters
    for i, data in tqdm.tqdm(enumerate(train_loader),
                             file = sys.stdout,
                             desc='Updating',
                             total=len(train_loader), 
                             leave=False):
        # get the inputs
        inputs, labels = data
        
        # Move them to the GPU
        inputs = togpu(inputs)
        labels = togpu(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)               # Predict
        loss = F.binary_cross_entropy(outputs, labels)   # Grade / Evaluate
        loss.backward()                     # Determine how each parameter effected the loss
        optimizer.step()                    # Update parameters 

        # print statistics
        running_loss += loss.item()
    running_loss /= len(train_loader)
    

    eval_loss = compute_eval_loss(net, test_loader)
    
    tend = time.time()
    
    # Save parameters
    torch.save(dict(epoch=epoch, 
                         loss=eval_loss,
                         parameters=net.state_dict(),
                         optimizer=optimizer.state_dict()),
                   'simplecnn-checkpoint.pth.tar')
    
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_epoch = epoch
        shutil.copyfile('simplecnn-checkpoint.pth.tar', 'simplecnn-best.pth.tar')
        
    print("Epoch {: 4}   loss: {: 2.5f}  test-loss: {: 2.5}  time: {}".format(epoch,
                                                                                running_loss,
                                                                                eval_loss,
                                                                                tend-tstart))
predictions = np.zeros((len(valid_data),17))
targets = np.zeros((len(valid_data),17))

for i  in tqdm.tnrange(len(valid_data)):
    for j in range(17): 
        set_trace()
        x, t = valid_data[i]
        p = tocpu(net(togpu(x[None,...]))).argmax(1)[0]
        predictions[i][j] = p
        targets[i][j] = t[i][j]

%debug
from sklearn.metrics import classification_report
print(classification_report(targets, predictions))
%debug
