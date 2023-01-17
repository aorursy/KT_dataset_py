import torch 
import torch.nn as nn 
from PIL import Image, ImageFile
import numpy as np 
import os
import glob 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 
import json
import gc
from pathlib import Path
from tqdm import tqdm
train_dir = "/kaggle/input/imaterialist-fashion-2020-fgvc7/train"
img_files = glob.glob(os.path.join(train_dir, "*.jpg"))
idx = 0
pil_im = Image.open(img_files[idx])
%matplotlib inline 
print("file_name: ", img_files[idx])
plt.imshow(np.asarray(pil_im))
data_dir = Path("/kaggle/input/imaterialist-fashion-2020-fgvc7/")
img_dir =  Path("/kaggle/input/imaterialist-fashion-2020-fgvc7/train")
train_data = pd.read_csv(data_dir/"train.csv")
train_data.sample(5)
# _Start: to get label descriptions 
with open(data_dir/"label_descriptions.json", 'r') as file:
    label_desc = json.load(file)
# _End: to get label descriptions 
label_desc
# _Start: Classes and Attributes processing 
df_categories = pd.DataFrame(label_desc['categories'])
df_attributes = pd.DataFrame(label_desc['attributes'])
# _End: Classes and Attributes processing 

n_classes = len(label_desc['categories'])
n_attributes = len(label_desc['attributes'])

print('Classes: {0} \nAttributes: {1}'.
     format(str(n_classes), str(n_attributes)))
df_categories
df_attributes
df_categories.supercategory.unique()
df_attributes.supercategory.unique()
def show_images(size = 4, figsize = (12, 12)):
    #get the images
    image_ids = train_data['ImageId'].unique()[:size]
    images = []
    
    for image_id in image_ids:
        images.append(mpimg.imread('{0}/train/{1}.jpg'.format(data_dir, image_id)))
        
    count = 0
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
    for row in ax:
        for col in row:
            col.imshow(images[count])
            col.axis('off')
            count += 1
    plt.show()
    gc.collect()
show_images()
#Function to create mask
def create_mask(size):
    image_ids = train_data['ImageId'].unique()[:size] #get a number of images
    images_meta = [] #to be added in this array
    
    for image_id in image_ids:
        img = mpimg.imread('{0}/train/{1}.jpg'.format(data_dir, image_id))
        images_meta.append({
            'image': img,
            'shape': img.shape,
            'encoded_pixels': train_data[train_data['ImageId'] == image_id]['EncodedPixels'],
            'class_ids': train_data[train_data['ImageId'] == image_id]['ClassId']
        })
        
    masks = []
    
    for image in images_meta:
        shape = image.get('shape') #get via key
        encoded_pixels = list(image.get('encoded_pixels')) 
        class_ids = list(image.get('class_ids'))
        
        #Initialize numpy array with shape same as image size
        height, width = shape[:2] 
        mask = np.zeros((height, width)).reshape(-1) 
        # (-1) 'The new shape should be compatible with the original shape'
        # numpy allow us to give one of new shape parameter as -1 but not (-1, -1)).
        # It means that it is an unknown dimension and we want numpy to figure it out.
        # And numpy will figure this by looking at the 'length of the array and remaining
        # dimensions' and making sure it satisfies the above mentioned criteria
        
        #Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))      #split the pixels string
            pixel_starts = splitted_pixels[::2]                      #choose every second element
            run_lengths = splitted_pixels[1::2]                      #start from 1 with step size 2
            assert max(pixel_starts) < mask.shape[0]                 #make sure it is ok
            
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id 
        masks.append(mask.reshape((height, width), order = 'F'))
    
    return masks, images_meta
def plot_segmented_images(size = 4, figsize = (14, 14)):
    #First, create masks from given segments
    masks, images_meta = create_mask(size)
    
    #Plot images
    
    count = 0
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
    for row in ax:
        for col in row:
            col.imshow(images_meta[count]['image'])
            col.imshow(masks[count], alpha = 0.50)
            col.axis('off')
            count += 1
    plt.show()
    gc.collect()
plot_segmented_images()
images_data = train_data.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
dimensions_data = train_data.groupby('ImageId')['Height', 'Width'].mean()

images_data = images_data.join(dimensions_data, on='ImageId')
images_data.head(5)
print("Total images: ", len(images_data))
class Fashion2020dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transforms, df_csv:str ):  # _to load and preprocess for the dataset. 
        
        super().__init__()
        self.data_root = data_root         
        self.transforms = transforms 
        self.imgs = list(sorted(os.listdir(os.path.join(data_root, "train"))))
        
        # _Start: read .csv with pandas for DataFormat description 
        self.df_csv = pd.read_csv(os.path.join(data_root, df_csv))  
        self.image_ids = self.df_csv["ImageId"].unique() # to get all image names
        
        
         
    def __getitem__(self, idx): # _to get a specific item.                        
        
        imgID = self.image_ids[idx]       
        #imgID = self.imgs[idx]       
        
        print(f"Image loading: {imgID}")
        
        pil_im = Image.open("{0}/train/{1}.jpg".format(self.data_root, str(imgID)))
        img = np.asarray(pil_im)
        
        images_meta = {} # 
        images_meta.update({ "image":img,
                             "shape":img.shape, 
                             "encoded_pixels": self.df_csv[self.df_csv['ImageId'] == imgID]['EncodedPixels'],
                             "class_ids" : self.df_csv[self.df_csv['ImageId'] == imgID]['ClassId']                                   
                             })
            
        # _Start: create masks with decoding and bbox from them 
        masks = []     
        boxes = [] 

        shape = images_meta.get("shape")  # _get via key of dict() 
        encoded_pixels = list(images_meta.get("encoded_pixels"))
        class_ids = list(images_meta.get("class_ids"))
        print(class_ids)
            
        # _Initialze an empty array with the same shape as the image 
        height, width = shape[:2] 
        mask = np.zeros((height, width)).reshape(-1)
        # (-1) 'The new shape should be compatible with the original shape'
            
        pbarLoad = tqdm(zip(encoded_pixels, class_ids))
        for segment, (pixel_str, class_id) in enumerate(pbarLoad):
            pbarLoad.set_description(f"Loading encoded pixels...: {segment}" )
            splitted_pixels = list(map(int, pixel_str.split())) #split the pixels string
            pixel_starts = splitted_pixels[::2] #choose every second element
            run_lengths = splitted_pixels[1::2]  #start from 1 with step size 2
               
            assert max(pixel_starts) < mask.shape[0]  
            
            pbarDecode = tqdm(zip(pixel_starts, run_lengths))    
            for pixel_start, run_length in pbarDecode:
                pbarDecode.set_description(f"Decoding masks...: {pixel_start}" )
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id *4
                         
            
            mask = mask.reshape((height, width), order = 'F')
            masks.append(mask)
            
            # _Start: get bounding box coordinates from each mask 
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # _End: get bounding box coordinates from each mask 
            
            mask = np.zeros((height, width)).reshape(-1) # re-initialize 
        # _End: create masks with decoding 
        
        
        
        # _Start: convert everything into a torch.Tensor 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_ids = torch.as_tensor(class_ids, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)  
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])      
        
        iscrowd = torch.zeros(len(class_ids,), dtype=torch.int64) # suppose all instances are not crowd
        # _End: convert everything into a torch.Tensor
        
        
        target = {}
        target["boxes"] = boxes
        target["class_ids"] = class_ids
        target["masks"] = masks
        target["image_id"] = image_id 
        target["area"] = area
        target["iscrowd"] = iscrowd           
        
        
        if self.transforms is not None: 
            img, target = self.transforms(img, target)           
        
        return img, target
    
    def __len__(self):          # _to return the length of data samples in the dataset. 
        return len(self.imgs)
Dataloader = Fashion2020dataset(data_root= data_dir, transforms=None, df_csv="train.csv")
img, target = Dataloader.__getitem__(100)
target["image_id"]
target["iscrowd"]
len(target["masks"])
type(target["masks"][1])
target["class_ids"]
target["boxes"]
from matplotlib.patches import Rectangle

%matplotlib inline 
idx = 5      # _ change the number withing range 
xmin, ymin, xmax, ymax = target["boxes"][idx]

plt.imshow(img)
plt.imshow(target["masks"][idx], alpha=0.7)

plt.gca().add_patch(Rectangle((xmin,ymin),xmax-xmin  , ymax-ymin, linewidth=1,edgecolor='r',facecolor='none'))
from matplotlib.patches import Rectangle

%matplotlib inline 
idx = 3      # _ change the number withing range 
xmin, ymin, xmax, ymax = target["boxes"][idx]

plt.imshow(img)
plt.imshow(target["masks"][idx], alpha=0.7)

plt.gca().add_patch(Rectangle((xmin,ymin),xmax-xmin  , ymax-ymin, linewidth=1,edgecolor='r',facecolor='none'))
