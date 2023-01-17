#You have to install torch 1.6, Fastai >=2.0.0 version.



!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html



#Upgrade kornia and allennlp version since current version does not support torch 1.6



!pip install --upgrade kornia

!pip install allennlp==1.1.0.rc4



#Install/upgrade fastai package



!pip install --upgrade fastai

#Load the libraries and verify the versions



import torch

print(torch.__version__)

print(torch.cuda.is_available())



import fastai

print(fastai.__version__)



from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images' #downloading and extracting images from the fast.ai datasets collection



def is_cat(x): return x[0].isupper() #function for grouping images after verifying the labels. Lable determines the type of image 

dls = ImageDataLoaders.from_name_func(

    path, get_image_files(path), valid_pct=0.2, seed=42,

    label_func=is_cat, item_tfms=Resize(224)) #define the type of dataset, validation percent and transform the images



learn = cnn_learner(dls, resnet34, metrics=error_rate) #create CNN for training the images, using resnet34 architecture and validate on the error_rate.

learn.fine_tune(1) #fit the model, in this case fine tune the model (2 epochs) since pretrained CNN is used
from PIL import Image



imagecat = Image.open("../input/catimage/manja-vitolic-gKXKBY-C-Dk-unsplash.jpg")

imagecat
from PIL import Image



imagedog = Image.open("../input/dogimage/josephine-menge-h7VBJRBcieM-unsplash.jpg")

imagedog
#convert JpegImageFile into numpy array

import numpy as np

imgcat=np.asarray(imagecat)

imgdog=np.asarray(imagedog)
#img = PILImage.create(uploader.data[0]) - In the lesson a widget is used to upload the image. so this line I have commented as I am using different image

is_cat,_,probs = learn.predict(imgcat)

print(f"Is this a cat?: {is_cat}.")

print(f"Probability it's a cat: {probs[1].item():.6f}")
is_cat,_,probs = learn.predict(imgdog)

print(f"Is this a cat?: {is_cat}.")

print(f"Probability it's a cat: {probs[1].item():.6f}")