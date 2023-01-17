import os

import xml.etree.ElementTree as ET



import torch

import torchvision



# for testing only

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
!cat ../input/annotation/Annotation/n02085620-Chihuahua/n02085620_10074
# This loader will use the underlying loader plus crop the image based on the annotation

def doggo_loader(path):

    img = torchvision.datasets.folder.default_loader(path) # default loader

    

    # Get bounding box

    annotation_basename = os.path.splitext(os.path.basename(path))[0]

    annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))

    annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)

    tree = ET.parse(annotation_filename)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox')

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

    bbox = (xmin, ymin, xmax, ymax)

    

    # return cropped image

    return img.crop(bbox)





# The dataset (example)

dataset = torchvision.datasets.ImageFolder(

    '../input/all-dogs/',

    loader=doggo_loader, # THE CUSTOM LOADER

    transform=torchvision.transforms.Compose([

        torchvision.transforms.Resize(64),

        torchvision.transforms.CenterCrop(64),

        torchvision.transforms.ToTensor(),

    ]) # some transformations, add your data preprocessing here

)
# Check that it all loads without a bug

for i in tqdm(range(len(dataset))):

    _ = dataset[i]

print('Ok.')
# Check that we get only the CUTE DOGS OH YES WHOS THE GOOD DOGGO ITS YOU

n = 10

_, axes = plt.subplots(figsize=(4*n, 4*n), ncols=n, nrows=n)

for i, ax in enumerate(axes.flatten()):

    ax.imshow(dataset[i][0].permute(1, 2, 0).detach().numpy())

plt.show()
# benchmark this loader vs vanilla



dataset_vanilla = torchvision.datasets.ImageFolder(

    '../input/all-dogs/',

    transform=torchvision.transforms.Compose([

        torchvision.transforms.Resize(64),

        torchvision.transforms.CenterCrop(64),

        torchvision.transforms.ToTensor(),

    ])

)
%%timeit

dataset[0]
%%timeit

dataset_vanilla[0]