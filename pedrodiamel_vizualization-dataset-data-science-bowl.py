import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import PIL.Image
import scipy.misc
import cv2
import skimage.color as skcolor
import skimage.util as skutl
import random
import csv
import matplotlib.pyplot as plt

#Class dsxbProvide
class dsxbProvide(object):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''    
    
    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        id_file_name,
        folders_image='images',
        folders_masks='masks',
        ):
        '''
        Factory function that create an instance of dsxbProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder, id_file_name, folders_image, folders_masks)
        provide._load_folders();
        return provide
    
    def __init__(self,
        base_folder,    
        sub_folder,     
        id_file_name,
        folders_image='images',
        folders_masks='masks',
        ):
        self.base_folder     = base_folder
        self.sub_folders     = sub_folder
        self.folders_image   = folders_image
        self.folders_masks   = folders_masks
        self.id_file_name    = id_file_name
        self.index           = 0
        self.data            = []        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        image_pathname = self.data[i][1];  
        masks_pathname = self.data[i][2];         
        #load image
        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)        
        #load mask
        label = np.zeros( (image.shape[0], image.shape[1], len(masks_pathname)  ) )
        for k,pathname in enumerate(masks_pathname):
            mask = np.array(self._loadimage(pathname), dtype=np.uint8)
            label[:,:,k] = mask*(k+1)         
        return image, label
           

    def _load_folders(self):
        '''
        load file patch for disk
        '''        
        self.data = []                
        folder_path = os.path.join(self.base_folder, self.sub_folders )
        in_id_path = os.path.join(self.base_folder, self.id_file_name )
        ext = 'png'        
        with open(in_id_path) as csvfile:            
            id_files = csv.reader(csvfile) 
            head = True
            id_current = ''
            data = {}
            for row in id_files:                 
                if head: head=False; continue;
                id_data = row[0]
                rl_code = [int(x) for x in row[1].split(' ')] 
                if id_data != id_current:                    
                    id_current  = id_data                    
                    images_path = os.path.join(folder_path, id_data, self.folders_image)
                    masks_path  = os.path.join(folder_path, id_data, self.folders_masks) 
                    masks_files = [ os.path.join(masks_path,f) \
                                   for f in sorted(os.listdir(masks_path)) \
                                   if f.split('.')[-1] == ext ];
                    image_file  = os.path.join(images_path, '{}.{}'.format(id_data,ext) )                    
                    data[id_current] = (id_current, image_file, masks_files, [], [])   
                data[id_current][3].append(rl_code)
        # to array        
        self.data = [ v for k,v in data.items()]


    def _loadimage(self, pathname):
        '''
        Load image using pathname
        '''
        if os.path.exists(pathname):
            try:
                image = PIL.Image.open(pathname)
                image.load()
            except IOError as e:
                raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message) ) 
        else:
            raise ValueError('"%s" not found' % pathname)
            
        return image;

pathdataset = '../input'
namedataset = '.'
metadata = 'stage1_train_labels.csv'
pathname = os.path.join(pathdataset, namedataset);
pathmetadata = os.path.join(pathdataset, namedataset, metadata)

base_folder = pathname
sub_folder =  'stage1_train'
id_file_name = metadata
folders_image='images'
folders_masks='masks'

dataloader = dsxbProvide.create(
    base_folder, 
    sub_folder, 
    id_file_name, 
    folders_image, 
    folders_masks
    )

print('Load dataset')
print('Total:',len(dataloader))
print(':)')

def torgb(im):
    if len(im.shape)==2:
        im = np.expand_dims(im, axis=2) 
        im = np.concatenate( (im,im,im), axis=2 )
    return im
        
def setcolor(im, mask, color):
    
    tmp=im.copy()
    tmp=np.reshape( tmp, (-1, im.shape[2])  )   
    mask = np.reshape( mask, (-1,1))      
    tmp[ np.where(mask>0)[0] ,:] = color
    im=np.reshape( tmp, (im.shape)  )
    return im

def lincomb(im1,im2,mask, alpha):
    
    #im = np.zeros( (im1.shape[0], im1.shape[1], 3) )
    im = im1.copy()    
    
    row, col = np.where(mask>0)
    for i in range( len(row) ):
        r,c = row[i],col[i]
        #print(r,c)
        im[r,c,0] = im1[r,c,0]*(1-alpha) + im2[r,c,0]*(alpha)
        im[r,c,1] = im1[r,c,1]*(1-alpha) + im2[r,c,1]*(alpha)
        im[r,c,2] = im1[r,c,2]*(1-alpha) + im2[r,c,2]*(alpha)
    return im

def makebackgroundcell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imlabel = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        imlabel = setcolor(imlabel,mask,color[:3])
    return imlabel

def makeedgecell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imedge = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        mask = mask.astype(np.uint8)
        _,contours,_ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        for cnt in contours: cv2.drawContours(imedge, cnt, -1, color[:3], 1)
    return imedge

def makeimagecell(image, labels, alphaback=0.3, alphaedge=0.3):
    
    imagecell = image.copy()
    imagecell = imagecell - np.min(imagecell)
    imagecell = imagecell / np.max(imagecell)    
    imagecell = torgb(imagecell)
     
    mask  = np.sum(labels, axis=2)
    imagecellbackground = makebackgroundcell(labels)
    imagecelledge = makeedgecell(labels)
    maskedge = np.sum(imagecelledge, axis=2)
    
    imagecell = lincomb(imagecell,imagecellbackground, mask, alphaback )
    imagecell = lincomb(imagecell,imagecelledge, maskedge, alphaedge )
            
    return imagecell
       

#k = np.random.randint( len(dataloader) )
image, labels = dataloader[ 538 ]

plt.figure( figsize=(16,16))
plt.subplot(311)
imagecell = makeimagecell(image, labels, alphaback=0.3, alphaedge=0.3)
plt.imshow( imagecell )
plt.title('Image+Background+Edge')
plt.axis('off')
plt.ioff()
plt.subplot(312)
imagecell = makeimagecell(image, labels, alphaback=1.0, alphaedge=0.0)
plt.imshow( imagecell )
plt.title('Image+Background')
plt.axis('off')
plt.ioff()
plt.subplot(313)
imagecell = makeimagecell(image, labels, alphaback=0.0, alphaedge=1.0)
plt.imshow( imagecell )
plt.title('Image+Edge')
plt.axis('off')
plt.ioff()

plt.show() 
def display_samplas(dataloader, row=3,col=3, alphaback=0.3, alphaedge=0.2):
    """
    Display random data from dataset
    For debug only
    """
    fig, ax = plt.subplots(row, col, figsize=(8,8), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})    
    for i in range(row):
        for j in range(col):            
            k = np.random.randint( len(dataloader) )
            image, labels = dataloader[ k ]
            imagecell = makeimagecell(image, labels, alphaback=alphaback, alphaedge=alphaedge)            
            ax[i,j].imshow(imagecell)
            ax[i,j].set_title('Image Idx: %d' % (k,))
    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()
    plt.show()

np.random.seed(4)
display_samplas(dataloader,row=4, col=4)
