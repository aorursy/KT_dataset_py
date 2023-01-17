# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

#torch.hub.list('pytorch/vision', force_reload=True)
#model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
#dir(model)
import torch

model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)

model.eval()
# Download an example image from the pytorch website

import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")

try: urllib.URLopener().retrieve(url, filename)

except: urllib.request.urlretrieve(url, filename)
#filename="/kaggle/input/pennfudanped/PNGImages/FudanPed00032.png"
from PIL import Image

input_image = Image.open(filename)
input_image
#writing dataloaders

#from torchvision import transforms, datasets



#data_transform = transforms.Compose([

 #       transforms.RandomSizedCrop(224),

  #      transforms.RandomHorizontalFlip(),

   #     transforms.ToTensor(),

    #    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                         #    std=[0.229, 0.224, 0.225])

    #])

#natural_dataset = datasets.ImageFolder(root='/kaggle/input/natural-images/data/natural_images/',

 #                                          transform=data_transform)

#dataset_loader = torch.utils.data.DataLoader(natural_dataset,

 #                                            batch_size=4, shuffle=True,

  #                                           num_workers=4)
#len(natural_dataset)
#filename1="/kaggle/input/natural-images/data/natural_images/cat/cat_0844.jpg"

filename1="/kaggle/input/natural-images/data/natural_images/motorbike/motorbike_0598.jpg"

input_image1 = Image.open(filename1)

input_image1
# create a color pallette, selecting a color for each class

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

colors = torch.as_tensor([i for i in range(21)])[:, None] * palette

colors = (colors % 255).numpy().astype("uint8")



# plot the semantic segmentation predictions of 21 classes in each color

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)

r.putpalette(colors)



import matplotlib.pyplot as plt

plt.imshow(r)

# plt.show()
filename2="/kaggle/input/natural-images/data/natural_images/person/person_0301.jpg"

input_image2 = Image.open(filename2)

input_image2
# create a color pallette, selecting a color for each class

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

colors = torch.as_tensor([i for i in range(21)])[:, None] * palette

colors = (colors % 255).numpy().astype("uint8")



# plot the semantic segmentation predictions of 21 classes in each color

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image2.size)

r.putpalette(colors)



import matplotlib.pyplot as plt

plt.imshow(r)

# plt.show()
# sample execution (requires torchvision)

from PIL import Image

from torchvision import transforms

input_image = Image.open(filename1)

preprocess = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



input_tensor = preprocess(input_image)

input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model



# move the input and model to GPU for speed if available

if torch.cuda.is_available():

    input_batch = input_batch.to('cuda')

    model.to('cuda')



with torch.no_grad():

    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)

# sample execution (requires torchvision)

from PIL import Image

from torchvision import transforms

input_image2 = Image.open(filename2)

preprocess = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



input_tensor = preprocess(input_image2)

input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model



# move the input and model to GPU for speed if available

if torch.cuda.is_available():

    input_batch = input_batch.to('cuda')

    model.to('cuda')



with torch.no_grad():

    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)
