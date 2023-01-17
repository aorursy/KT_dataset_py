'''# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in sorted(os.walk('/kaggle/input')):
    for filename in filenames:
        #if filename.endswith('.jpg'):
        #    break
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session'''
#There was not enough time to train this full model, but here is what was done.
'''import torch

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
'''
import numpy as np  
import pandas as pd
import torch
import os
from torch import nn
from torch import optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
'''import cv2
import re
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, image
from keras.applications.resnet50 import preprocess_input, decode_predictions, resnet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential 
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K'''
test_dir = '/kaggle/input/iwildcam-2020-fgvc7/test/'
train_dir = '/kaggle/input/iwildcam-2020-fgvc7/train/'
train_filenames = []
test_filenames = []
train_filepaths = []
test_filepaths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(dirname)
        #print(filename)
        #print(dirname==train_dir)
        #print(dirname==test_dir)
        #break
        if dirname.endswith('train'):
            train_filenames.append(filename[:-4])
            #train_filepaths.append(os.path.join(dirname, filename))
        if dirname.endswith('test'):
            test_filenames.append(filename[:-4])
            #test_filepaths.append(os.path.join(dirname, filename))
#train_filenames
len(test_filenames)
'''import os
from PIL import Image 
fn_nopen_train = []
#im = Image.open('/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg')

for filename in train_filepaths:
    try:
        #path = r'/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg'
        Image.open(filename)
    except:
        print('didnt open ' + filename)
        fn_nopen_train.append(filename)
'''
'''
fn_nopen_test = []
#im = Image.open('/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg')

for filename in test_filepaths:
    try:
        #path = r'/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg'
        Image.open(filename)
    except:
        print('didnt open ' + filename)
        fn_nopen_test.append(filename)'''
#train_filepaths 
#test_filepaths
'''fn_nopen_train = ['/kaggle/input/iwildcam-2020-fgvc7/train/883572ba-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/8792549a-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/99136aa6-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/8f17b296-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/896c1198-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg']'''
nopen_train = ['/kaggle/input/iwildcam-2020-fgvc7/train/883572ba-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/8792549a-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/99136aa6-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/8f17b296-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/896c1198-21bc-11ea-a13a-137349068a90.jpg',
 '/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg']
nopen_train_id = [i[40:-4] for i in nopen_train]
nopen_train_id
#train_filenames
#len(train_filenames)

#len(test_filenames)
import json

with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json', 'r', errors='ignore') as f:
    train_annotations = json.load(f)
samp_sub = pd.read_csv('/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv')
samp_sub
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json', 'r', errors='ignore') as f:
    test_information = json.load(f)
#with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json', 'r', errors='ignore') as f:
#    megadetector_results = json.load(f)
#train_annotations.keys()
#test_information.keys()
#megadetector_results.keys()
train_ann = pd.DataFrame(train_annotations['annotations'])
#train_ann[train_ann['image_id'].isin(nopen_train_id)]

#train_allopen = train_ann.drop(train_ann[train_ann['image_id'].isin(nopen_train_id)].index)
#train_allopen
#train_fn_allopen =   [x for x in train_filenames if x not in nopen_train_id]
#len(train_fn_allopen)
#train_ann_image_index = train_allopen.set_index('image_id')
train_ann_image_index = train_ann.set_index('image_id')
train_ann_image_index
#train_ann_filename_order = train_ann_image_index.loc[train_fn_allopen]
train_ann_filename_order = train_ann_image_index.loc[train_filenames]
train = train_ann_filename_order.reset_index()
train
train_y = train['category_id'].values
train['category_id'].describe()
'''def convert_to_tensor(path_img):
    img = image.load_img(path_img, target_size = (224,224))
    img_arr = image.img_to_array(img)
    return np.expand_dims(img_arr, axis = 0)'''
'''def convert_all_tensor(paths_imgs):
    tensor_list = [convert_to_tensor(i) for i in paths_imgs]
    return np.vstack(tensor_list)'''
#train_tensors = convert_all_tensor(train_filepaths).astype('float64')/255
#test_tensors = convert_all_tensor(test_filepaths).astype('float64')/255
'''import os
from PIL import Image 
fn_nopen_train = []
#im = Image.open('/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        try:
            #path = r'/kaggle/input/iwildcam-2020-fgvc7/train/87022118-21bc-11ea-a13a-137349068a90.jpg'
            Image.open(train_dir + filename)
        except:
            print('didnt open' + train_dir + filename)
            fn_nopen_train.append(train_dir+ filename)'''
            
#open(path, 'rb')
#pil_loader(path)
#print(PILLOW_VERSION)
import torch.utils.data as data

from PIL import Image

import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        #d = os.path.join(dir, target)
        #if not os.path.isdir(d):
        #    continue

        #for root, _, fnames in sorted(os.walk(d)):
         #   for fname in sorted(fnames):
        if is_image_file(target):
            path = os.path.join(dir, target)
            item = (path,0) #, class_to_idx[target])
            images.append(item)
            #images.append(path) #avoid subfolders

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:        
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        try:
            img = self.loader(path)
        except:
            img = self.loader(train_dir+'8ccf922e-21bc-11ea-a13a-137349068a90.jpg')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target #no targets given, so none returned


    def __len__(self):
        return len(self.imgs)
#[d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
'''for target in sorted(os.listdir(train_dir)):
    print(target)
    break'''
'''for target in sorted(os.listdir(train_dir)):
    d = os.path.join(train_dir, target)
    if not os.path.isdir(d):
        print('y')
    print(d)
    break'''
'''classes = [d for d in os.listdir('/kaggle/input/iwildcam-2020-fgvc7/') if os.path.isdir(os.path.join('/kaggle/input/iwildcam-2020-fgvc7/', d))]
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
images = []
td = os.path.expanduser('/kaggle/input/iwildcam-2020-fgvc7/')
for target in sorted(os.listdir(td)):
    d = os.path.join(td, target)
    if not os.path.isdir(d):
        continue

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
                #images.append(path)
            break
        break       
    break

print(images)              '''  
'''classes = [d for d in os.listdir('/kaggle/input/iwildcam-2020-fgvc7/') if os.path.isdir(os.path.join('/kaggle/input/iwildcam-2020-fgvc7/', d))]
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
images = []
td = os.path.expanduser('/kaggle/input/iwildcam-2020-fgvc7/')
for target in sorted(os.listdir(td)):
    d = os.path.join(td, target)
    if is_image_file(fname):
        path = os.path.join(root, fname)
        images.append(path)
      
    break

print(images)'''




train_torch_y = torch.tensor(train_y, dtype=torch.long ).view(2247, 97)
#train_torch_y
#train_cat = pd.DataFrame(train_annotations['categories'])
#train_cat
#train_imgs = pd.DataFrame(train_annotations['images'])
#train_imgs
#train_annotations['info']
#test_information['info']
#test_imgs = pd.DataFrame(test_information['images'])
#test_imgs
#test_cat = pd.DataFrame(test_information['categories'])
#test_cat
#test_cat['count'].sum()
#megadetector_cat = pd.Series(megadetector_results['detection_categories'])
#megadetector_cat
#megadetector_imgs = pd.DataFrame(megadetector_results['images'])
#megadetector_imgs
#megadetector_results['info']

'''def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)'''

data_transforms_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])



image_datasets_train =  ImageFolder(train_dir, transform=data_transforms_train)



dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=97)#,collate_fn=collate_fn)
#next(iter(dataloaders_train))[0].shape
data_transforms_test = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

image_datasets_test = ImageFolder(test_dir, transform=data_transforms_test)

dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=82)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg16_bn(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 572),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
model.to(device)
epochs = 2
steps = 0
running_loss = 0
print_every = 100
for epoch in range(epochs):
    for inputs, labels in zip(dataloaders_train,train_torch_y):
        steps += 1
        inputs, labels = inputs[0].to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. ")
            running_loss = 0
     
            if steps % 200 == 0:
                break #to get out of loop before memory is full

model.eval()
top_c = []
top_probs = []
with torch.no_grad():
    for inputs, labels in dataloaders_test:
        inputs  = inputs.to(device) 
        logps = model.forward(inputs)
                           
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        top_c.append(top_class)
        #top_probs.append(top_p)
model.train()
#top_c
topc_list = [torch.Tensor.cpu(i).numpy().tolist() for i in top_c]
from functools import reduce
topc_list1 = reduce(lambda x,y: x+y, topc_list)
topc_list2 = reduce(lambda x,y: x+y, topc_list1)
len(topc_list2)
type(topc_list2)
#Image.open(train_dir+'86760c00-21bc-11ea-a13a-137349068a90.jpg')
topc_ser = pd.Series(topc_list2)
#topc_ser


sub = samp_sub.copy()
sub 
sub_image_index = sub.set_index('Id')
#sub_image_index.index.values
test_fn_sub = [i for i in test_filenames if i in sub_image_index.index.values]
not_in_sub = [i for i in test_filenames if i not in sub_image_index.index.values]
len(not_in_sub)
extra_index = [test_filenames.index(i) for i in not_in_sub]
topc_ser_drop_extra = topc_ser.drop(extra_index).reset_index()
#topc_ser_drop_extra
len(test_fn_sub)
sub_filename_order = sub_image_index.loc[test_fn_sub]
#sub_filename_order
sub_filename_order['Category'] = topc_ser_drop_extra[0].values
#topc_ser_drop_extra[0].values
sub_filename_order_cat = sub_filename_order.reset_index()
#sub_filename_order_cat
sub_filename_order_cat.to_csv('submission.csv', index=False)
