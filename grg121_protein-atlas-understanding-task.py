%matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import copy


from PIL import Image

DATASET_SIZE = 3500
BATCH_SIZE = 500
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
import torch
import random
import numpy as np
from skimage import filters
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from skimage import io, transform
from skimage.util import img_as_ubyte
from skimage.transform import resize
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
classes = np.arange(0,28)
mlb = MultiLabelBinarizer(classes)
mlb.fit(classes)

class HumanProteinDataset(Dataset):

    def __init__(self, csv_file,transform=None, test=False):
        self.test = test
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        if not test:
            self.path = train_path
            self.df['Targets'] = self.df['Target'].map(lambda x: list(map(int, x.strip().split())))

        else:
            self.path = test_path
            
    def load_image(path, image_id):
        images = np.zeros(shape=(256,256,4))
        r = resize(img_as_ubyte(io.imread(os.path.join(path+image_id+"_red.png"),as_gray=True)),(W,H))
        g = resize(img_as_ubyte(io.imread(os.path.join(path+image_id+"_green.png"),as_gray=True)),(W,H))
        b = resize(img_as_ubyte(io.imread(os.path.join(path+image_id+"_blue.png"),as_gray=True)),(W,H))
        y = resize(img_as_ubyte(io.imread(os.path.join(path+image_id+"_yellow.png"),as_gray=True)),(W,H))

        images[:,:,0] = np.asarray(r)
        images[:,:,1] = np.asarray(g)
        images[:,:,2] = np.asarray(b)
        images[:,:,3] = np.asarray(y)

        return images
            
    def __getitem__(self, idx):
        image = HumanProteinDataset.load_image(self.path, self.df['Id'].iloc[idx])
        sample = {'image': image}

        if not self.test:
            target = np.array(self.df['Targets'].iloc[idx])
            target = mlb.transform([target])
            sample['Target'] = target
        else:
            sample['Id'] = self.df['Id'].iloc[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def IndexOfOne(self, targets):
        fdf = self.df[self.df['Target'] == targets]
        idx = random.randint(0,fdf.shape[0])
        return fdf.index[idx]
        
    
    def __len__(self):
        return self.df.shape[0]
    
    def shape(self):
        return self.df.shape
    

dataset = HumanProteinDataset(train_csv_path)
def Show(sample):
    f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(25,15), sharey=True)

    title = ''
    
    labels = sample['Target'][0]
                
    for i, label in enumerate(LABELS):
        if labels[i] == 1:
            if title == '':
                title += label
            else:
                title += " & " + label
                
    rgb = np.zeros([W,H,3])
    
    rgb[:,:,0] = sample['image'][:,:,0]
    rgb[:,:,1] = sample['image'][:,:,3] # green channel will be the yellow one
    rgb[:,:,2] = sample['image'][:,:,2]
    
    protein = sample['image'][:,:,1]
    
    ax2.imshow(rgb)
    ax2.set_title('Reference')
    ax1.imshow(protein)
    ax1.set_title('Protein')
    
    protein = filters.gaussian(protein, sigma= 0.8)
    protein = (protein - protein.min())/(protein.max() - protein.min())
    #protein = protein > 0.95 * protein.mean()
    
    rgb[:,:,0] *= protein
    rgb[:,:,1] *= protein
    rgb[:,:,2] *= protein
    
    ax3.imshow(rgb)
    ax3.set_title('Reference (filterd)')
    ax4.imshow(protein)
    ax4.set_title('Protein mask')
    f.suptitle(title, fontsize=15, y=0.68)
for i in range(8):
    idx = dataset.IndexOfOne('0')
    Show(dataset[idx])

for i in range(8):
    idx = dataset.IndexOfOne('25')
    Show(dataset[idx])
for i in range(8):
    idx = dataset.IndexOfOne('25 0')
    Show(dataset[idx])
for i in range(8):
    idx = dataset.IndexOfOne('23')
    Show(dataset[idx])