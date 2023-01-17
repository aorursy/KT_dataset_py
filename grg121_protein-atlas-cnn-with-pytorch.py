%matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import copy


from PIL import Image

DATASET_SIZE = 8000
BATCH_SIZE = 200
W = H = 256

train_path = '../input/train/'
test_path = '../input/test/'

LABEL_MAP = {
0: "Nucleoplasm" ,
1: "Nuclear membrane"   ,
2: "Nucleoli"   ,
3: "Nucleoli fibrillar center",   
4: "Nuclear speckles"   ,
5: "Nuclear bodies"   ,
6: "Endoplasmic reticulum"   ,
7: "Golgi apparatus"  ,
8: "Peroxisomes"   ,
9:  "Endosomes"   ,
10: "Lysosomes"   ,
11: "Intermediate filaments"  , 
12: "Actin filaments"   ,
13: "Focal adhesion sites"  ,
14: "Microtubules"   ,
15: "Microtubule ends"   ,
16: "Cytokinetic bridge"   ,
17: "Mitotic spindle"  ,
18: "Microtubule organizing center",  
19: "Centrosome",
20: "Lipid droplets"   ,
21: "Plasma membrane"  ,
22: "Cell junctions"   ,
23: "Mitochondria"   ,
24: "Aggresome"   ,
25: "Cytosol" ,
26: "Cytoplasmic bodies",
27: "Rods & rings"}

LABELS = []

for label in LABEL_MAP.values():
    LABELS.append(label)
    
train_csv_path = '../input/train.csv'
df = pd.read_csv(train_csv_path)

TRAINING_SAMPLES = df.shape[0]

print("we have " + str(TRAINING_SAMPLES) + " different samples")
print("And there are "+  str(len(df.Target.unique())) + " different combinations of labels in our dataset")

import seaborn as sns
sns.set(style="dark")

n = 20

values = df['Target'].value_counts()[:n].keys().tolist()
counts = df['Target'].value_counts()[:n].tolist()

plt.figure(figsize=(6,6))
pal = sns.cubehelix_palette(n, start=2, rot=0, dark=0, light=.75, reverse=True)
g = sns.barplot(y=counts, x=values, palette=pal)
g.set_title(str(n)+" MOST COMMON LABEL COMBINATIONS")
g.set_xticklabels(g.get_xticklabels(),rotation=90);
from PIL import Image

def load_image(basepath, image_id):
    images = np.zeros(shape=(256,256,4))
    r = Image.open(basepath+image_id+"_red.png").resize((256,256))
    g = Image.open(basepath+image_id+"_green.png").resize((256,256))
    b = Image.open(basepath+image_id+"_blue.png").resize((256,256))
    y = Image.open(basepath+image_id+"_yellow.png").resize((256,256))

    images[:,:,0] = np.asarray(r)
    images[:,:,1] = np.asarray(g)
    images[:,:,2] = np.asarray(b)
    images[:,:,3] = np.asarray(y)
    
    return images
targets = df['Target'].value_counts().keys()
counts = df['Target'].value_counts().values

how_many = counts/TRAINING_SAMPLES*DATASET_SIZE

# at least one example of each possible combination of labels..
how_many = how_many.astype('int')+1

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from skimage import io, transform
from sklearn.preprocessing import MultiLabelBinarizer
classes = np.arange(0,28)
mlb = MultiLabelBinarizer(classes)
mlb.fit(classes)

class HumanProteinDataset(Dataset):

    def __init__(self, csv_file,transform=None, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            test (Boolean): the csv no contains labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test = test
        self.complete_df = pd.read_csv(csv_file)
        
        if not test:
            self.path = train_path
            self.loadData()
        else:
            self.path = test_path
            self.df = self.complete_df
            
        self.transform = transform
        
    def CreateDummyVariables(self):
        self.complete_df['Targets'] = self.complete_df['Target'].map(lambda x: list(map(int, x.strip().split())))
            
    def loadData(self):
        self.CreateDummyVariables()
        self.df = pd.DataFrame(columns=['Id','Target'])
        for i, target in enumerate(targets):
            fdf = self.complete_df[self.complete_df['Target'] == target]
            sample = fdf.sample(n=how_many[i], replace=False)
            self.df = self.df.append(sample)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
            
    def __getitem__(self, idx):
        
        image = load_image(self.path, self.df['Id'].iloc[idx])
        
        sample = {'image': image}

        if not self.test:
            target = np.array(self.complete_df['Targets'].iloc[idx])
            target = mlb.transform([target])
            sample['target'] = target
        
        else:
            sample['Id'] = self.df['Id'].iloc[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.df.shape[0]
    
    def shape(self):
        return self.df.shape
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image = sample['image']/255.0
        
        totensor = transforms.ToTensor()
        
        ret = {'image': totensor(image)}
        
        if "target" in sample.keys():
            target = sample['target'][0]
            ret['target'] = target
        else:
            ret['Id'] = sample['Id']
                  
        return ret
dataset = HumanProteinDataset(train_csv_path, transform=ToTensor())
def Show(sample):
    f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(25,25), sharey=True)

    title = ''
    
    labels =sample['target']
                
    for i, label in enumerate(LABELS):
        if labels[i] == 1:
            if title == '':
                title += label
            else:
                title += " & " + label
            
    ax1.imshow(sample['image'][0,:,:],cmap="hot")
    ax1.set_title('Red')
    ax2.imshow(sample['image'][1,:,:],cmap="copper")
    ax2.set_title('Green')
    ax3.imshow(sample['image'][2,:,:],cmap="bone")
    ax3.set_title('Blue')
    ax4.imshow(sample['image'][3,:,:],cmap="afmhot")
    ax4.set_title('Yellow')
    f.suptitle(title, fontsize=20, y=0.62)
import random

idxs = random.sample(range(1, dataset.df.shape[0]), 3)

for idx in idxs:
    Show(dataset[idx])
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import math

def prepare_loaders():
    dataset.loadData()
    num_train = len(dataset)
    indices = list(range(num_train))
    val_size = int(0.45 * num_train) 

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=val_size, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    dataset_sizes = {}

    dataset_sizes['train'] = len(train_idx)
    dataset_sizes['val'] = len(validation_idx)
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,num_workers=0, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0,sampler=validation_sampler)

    dataloaders = {}

    dataloaders['train'] = train_loader
    dataloaders['val'] = validation_loader
    
    return (dataloaders, dataset_sizes)

dataloaders, dataset_sizes = prepare_loaders()

dataset.df.head()
# Wout = 1 + (Win - Kernel_size + 2Padding)/Stride

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(      #input: 4xWxH
            nn.Conv2d(4,8,5,1,2),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2), #output: 8xW/2xH/2
        )
        self.conv2 = nn.Sequential(      #input: 4xWxH
            nn.Conv2d(8,16,5,1,2),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2), #output: 16xW/4xH/4
        )
        self.drop_out = nn.Dropout()
        self.out1 = nn.Linear( int(16 * W/4 * H/4), 900)   # fully connected layer, output 28 classes
        self.out2 = nn.Linear( 900, 28)   # fully connected layer, output 28 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.drop_out(x)
        output = self.out1(x)
        output = self.out2(output)
        return output, x    # return x for visualization

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
# create a new subdataset for training
dataloaders, dataset_sizes = prepare_loaders()
losses = {}
accuracys = {}

losses['train'] = []
losses['val'] = []
accuracys['train'] = []
accuracys['val'] = []
def Train(model, epochs=10, criterion=nn.BCEWithLogitsLoss(reduction='sum'), optimizer= None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=0.04, betas=(0.9, 0.99))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("training with device: " + str(device))
    
    model.to(device)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0    
            running_corrects = 0.0
    
            for i, data in enumerate(dataloaders[phase], 0):            
                # get the inputs
                inputs, labels = data['image'], data['target']

                inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)[0]
                    preds = outputs > 0
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                 # statistics
                running_loss += loss.item() * inputs.size(0)
                labels = labels.data.byte()
                running_corrects += torch.sum((labels == preds).all(1))
                                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            losses[phase].append(epoch_loss)
            accuracys[phase].append(epoch_acc)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run_model(model,batch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = batch
    inputs = inputs.to(device,dtype=torch.float)
    out = model(inputs)
    out = out[0].cpu()
    return out
# model creation and initialization
cnn = CNN()
cnn.apply(init_weights)


# training
torch.cuda.empty_cache()
cnn = Train(cnn, epochs=10,  criterion=nn.BCEWithLogitsLoss(reduction='sum'), optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.99)))
# training
torch.cuda.empty_cache()
cnn = Train(cnn, epochs=10,  criterion=nn.BCEWithLogitsLoss(reduction='sum'), optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.99)))
dataloaders, dataset_sizes = prepare_loaders()
torch.cuda.empty_cache()
cnn = Train(cnn, epochs=10,  criterion=nn.BCEWithLogitsLoss(reduction='sum'), optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.99)))

plt.plot(np.arange(len(losses['train'])), losses['train'],label="train")
plt.plot(np.arange(len(losses['val'])), losses['val'], label="val")
plt.legend()
plt.title("loss by epoch")
plt.show()

plt.plot(np.arange(len(accuracys['train'])), accuracys['train'], label="train")
plt.plot(np.arange(len(accuracys['val'])), accuracys['val'], label="val")
plt.title("accuracy by epoch")
plt.legend()
plt.show()
def save(model, full = True, name="model"):
    if not full:
        torch.save(model.state_dict(), name+'_params.pkl')   # save only the parameters
    else:
        torch.save(model, name+'.pkl')  # save entire net

        
def restore_net(name="model"):
    model = torch.load(name+'.pkl')
    return model
    
save(cnn, name="cnn")
!ls
cnn = restore_net(name="cnn")

#!pip install dropbox
import dropbox
dropbox_path='/'
dbx=dropbox.Dropbox('Your access token')

# upload model to dropbox

file_name='cnn.pkl'
with open(file_name, 'rb') as f:
    dbx.files_upload(f.read(),dropbox_path+file_name,mute=True, mode=dropbox.files.WriteMode.overwrite)


# load model from dropbox
dbx.files_download_to_file(file_name,dropbox_path+file_name)
!ls
submit = pd.read_csv('../input/sample_submission.csv')

dataset_test = HumanProteinDataset(csv_file='../input/sample_submission.csv', transform=transforms.Compose([
    ToTensor()
]), test=True)

dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

ids = []
predictions = []

cnn = cnn.cuda()

for sample_batched in dataloader_test:
        out = run_model(cnn,sample_batched['image'])
        
        preds = []
        out = out.detach().numpy()
        for sample in out:
            p = ""
            for i,label in enumerate(sample):
                if label > 0:
                    p += " " + str(i)
                    print(p)
            if p == "":
                p = "0"
            else:
                p = p[1:]
            preds.append(p)

        ids += list(sample_batched['Id'])
        predictions += preds

    

print(predictions)
df = pd.DataFrame({'Id':ids,'Predicted':predictions})
df.to_csv('protein_classification.csv', header=True, index=False)

print(df)

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "protein_classification.csv"):  
    csv = df.to_csv( sep=',', encoding='utf-8', index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# create a link to download the dataframe
create_download_link(df)