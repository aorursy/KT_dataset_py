# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import torch

sns.set()

matplotlib.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


## data analysis
path = "../input"
metadata='train.csv'
image_train_path = os.path.join(path, 'train')
image_test_path = os.path.join(path, 'test')

train_data = pd.read_csv( os.path.join( path, metadata) )

print(train_data.head())
print('Total: ', len(train_data))

idx_to_class = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

class_to_idx = dict((v,k) for k,v in idx_to_class.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = idx_to_class[int(num)]
        row.loc[name] = 1
    return row

for key in idx_to_class.keys():
    train_data[idx_to_class[key]] = 0
    
train_data = train_data.apply(fill_targets, axis=1)
train_data.head()



target_counts = train_data.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
plt.show()

print(target_counts)

## load dataset
path = "../input"
metadata='train.csv'
image_train_path = os.path.join(path, 'train')
image_test_path = os.path.join(path, 'test')

train_data = pd.read_csv( os.path.join( path, metadata) )

print(train_data.head())
print('Total: ', len(train_data))
def fill_targets(row):
    target = np.array(row.Target.split(" ")).astype(np.int)
    p = np.zeros( 28 )
    p[target] = 1 #1/len(target)
    row.Target = p
    return row

train_data = train_data.apply(fill_targets, axis=1)
train_data.head()

# https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

def open_grby(path, id): #a function that reads GRBY image
    suffs = ['green', 'red', 'blue','yellow']
    cvflag = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join( path, '{}_{}.png'.format(id, suff) ), cvflag).astype(np.float32)/255 
           for suff in suffs ]
    return np.stack(img, axis=-1)

def grby2rgb( image ):
    return np.stack( ( image[:,:,0], image[:,:,1]/2 + image[:,:,3]/2, image[:,:,2]/2 + image[:,:,3]/2  ), axis=-1 )


index = 0
image_id = train_data['Id'][index]
prob = train_data['Target'][index]
image_grby = open_grby( image_train_path, image_id )

print(image_grby.shape)
print(prob)

ips = np.where( prob>0 )[0]
print( [ '{}:{}'.format( idx_to_class[ip], prob[ip]) for ip in ips  ]  )

plt.figure( figsize=(22,8) )
plt.subplot(151)
plt.imshow( grby2rgb(image_grby) )
plt.axis('off')
plt.title('image grby (grb)')
for i,v in enumerate(['green', 'red', 'blue','yellow']):
    plt.subplot(1,5,i+2)
    plt.imshow( image_grby[:,:,i] )
    plt.axis('off')
    plt.title('image grby ({}-channel)'.format( v[0] ) )
plt.show()

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['figure.figsize'] = (12,19)

numRows = 9; numCols = 5

plt.figure()
for k in range(numRows*numCols):
    index = np.random.randint( len(train_data) )
    image_id = train_data['Id'][index]
    prob = train_data['Target'][index]
    image_grby = open_grby( image_train_path, image_id )    
    plt.subplot(numRows,numCols,k+1); 
    plt.imshow( grby2rgb( image_grby )  )
    plt.title( '{} ...'.format( image_id[:3] ) ); 
    plt.axis('off')




def open_grby( path, id): 
    '''a function that reads GRBY image'''
    suffs = ['green', 'red', 'blue','yellow']
    cvflag = cv2.IMREAD_GRAYSCALE    
    img = [cv2.imread(os.path.join( path, '{}_{}.png'.format(id, suff) ), cvflag).astype(np.float32)/255 
           for suff in suffs ]
    return np.stack(img, axis=-1)

def make_dataset( path, metadata, train=True):
    '''load file patch for disk
    '''
    data = pd.read_csv( os.path.join( path, metadata) )
    if train:
        def fill_targets(row):
            target = np.array(row.Target.split(" ")).astype(np.int)
            p = np.zeros( 28 )
            p[target] = 1 #1/len(target)
            row.Target = p
            return row
        data = data.apply(fill_targets, axis=1)
    return data

def grby2rgb( image ):
    return np.stack( ( image[:,:,0], image[:,:,1]/2 + image[:,:,3]/2, image[:,:,2]/2 + image[:,:,3]/2  ), axis=-1 )

class ATLASProvide( object ):
    '''Provide for ATLAS dataset
    '''
    @classmethod
    def create(
        cls, 
        path,
        train=True,
        folders_images='train',
        metadata='train.csv',
        ):
        '''
        Factory function that create an instance of ATLASProvide and load the data form disk.
        '''
        provide = cls(path, train, folders_images, metadata )
        return provide
    
    def __init__(self,
        path,        
        train=True,
        folders_images='train',
        metadata='train.csv',
        ):
        super(ATLASProvide, self).__init__( )        
        self.path     = os.path.expanduser( path )
        self.folders_images  = folders_images
        self.metadata        = metadata
        self.data            = []
        self.train           = train
        
        self.data = make_dataset( self.path, self.metadata, self.train )
        
    def __len__(self):
        return len(self.data)
        
    def getname(self, i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        return self.data['Id'][i]        

    def __getitem__(self, i):                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i        
        if self.train:           
            image_id = self.data['Id'][i]
            prob = self.data['Target'][i]
            image_grby = open_grby(  os.path.join(self.path, self.folders_images ), image_id )
            return image_id, grby2rgb(image_grby), prob
        else:
            image_id = self.data['Id'][i]
            image_grby = open_grby( os.path.join(self.path, self.folders_images ) , image_id )
            return image_id, grby2rgb(image_grby), 0

        
        
path = "../input"
metadata='train.csv' # train.csv, sample_submission.csv
folders_images='train' #train, test
train=True #True, False
dataset = ATLASProvide.create(path=path, train=train, folders_images=folders_images, metadata=metadata )
iD,image, prob = dataset[ np.random.randint( len(dataset) ) ]

print( len(dataset) )     
print( iD )
print( prob )

plt.figure( figsize=(8,8) )
plt.imshow( image )
plt.axis('off')
plt.show()


!ls ..
# dataloader 

train = 'train'
validation = 'train'
test  = 'train'

class ATLASDataset(object):
    '''
    Management for Human Protein Atlas dataset
    '''
    def __init__(self, 
        path,   
        train=True,
        folders_images='train',
        metadata = 'train.csv',
        ext='png',
        transform=None,
        count=None, 
        num_channels=3,
        ):
        """Initialization       
        """            
           
        self.data = ATLASProvide.create( 
                path, 
                train,
                folders_images, 
                metadata,
                )
        
        self.transform = transform  
        self.count = count if count is not None else len(self.data)   
        self.num_channels = num_channels

    def __len__(self):
        return self.count
    
    def getname(self, idx):
        idx = idx % len(self.data)
        return self.data.getname(idx)

    def __getitem__(self, idx):   
        idx = idx % len(self.data)
        iD, image, prob = self.data[idx]
                
        #obj = ObjectImageTransform( image )
        #if self.transform: 
        #    obj = self.transform( obj )
        #image = obj.to_value()
        
        return iD, image, prob 
    

path = "../input"
metadata='train.csv' # train.csv, sample_submission.csv
folders_images='train' #train, test
train=True #True, False
dataset = ATLASDataset(path=path, train=train, folders_images=folders_images, metadata=metadata )
iD,image, prob = dataset[ np.random.randint( len(dataset) )  ]

print( len(dataset) )     
print( iD )
print( prob )

plt.figure( figsize=(8,8) )
plt.imshow( image )
plt.axis('off')
plt.show()




