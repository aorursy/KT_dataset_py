import numpy as np

import pandas as pd

from PIL import Image

import requests

from io import BytesIO



#!pip install matplotlib==3.3.2



import matplotlib.pyplot as plt



url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"

response = requests.get(url)

img = Image.open(BytesIO(response.content))

imgt = img.resize((img.size[0] // 4, img.size[1] // 4))

img_np = np.array(imgt)

plt.imshow(img_np)
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html



!pip install -U fastai

import fastai

print(fastai.__version__)



!pip install icevision

!pip install icedata



from icevision.all import *

import icedata



import torch

print(f'{torch.__version__} cuda is on: {torch.cuda.is_available()}')
plt.imshow(img_np)
device = "cpu"

device = "cuda"



class_map = icedata.datasets.pets.class_map()



model = faster_rcnn.model(num_classes=len(class_map))



url = "https://github.com/airctic/model_zoo/releases/download/m3/pets_faster_resnetfpn50.zip"

state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device(device))

model.load_state_dict(state_dict)



# good model inference

infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])

infer_ds = Dataset.from_images([img_np], infer_tfms)



batch, samples = faster_rcnn.build_infer_batch(infer_ds)

preds = faster_rcnn.predict(model=model, batch=batch)



infer_dl = faster_rcnn.infer_dl(infer_ds, batch_size=1)

samples, preds = faster_rcnn.predict_dl(model=model, infer_dl=infer_dl)



imgs = [sample["img"] for sample in samples]

show_preds(

    imgs=imgs,

    preds=preds,

    class_map=class_map,

    denormalize_fn=denormalize_imagenet,

    show=True,

    display_label=False,

)
# Load the PETS dataset

path = icedata.pets.load_data()



# Get the class_map, a utility that maps from number IDs to classs names

class_map = icedata.pets.class_map()



# Randomly split our data into train/valid

data_splitter = RandomSplitter([0.8, 0.2])



# PETS parser: provided out-of-the-box

parser = icedata.pets.parser(data_dir=path, class_map=class_map)

train_records, valid_records = parser.parse(data_splitter)



# shows images with corresponding labels and boxes

show_records(train_records[:6], ncols=3, class_map=class_map, show=True)



# Define transforms - using Albumentations transforms out of the box

train_tfms = tfms.A.Adapter(

    [*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()]

)

size=384

valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

# Create both training and validation datasets

train_ds = Dataset(train_records, train_tfms)

valid_ds = Dataset(valid_records, valid_tfms)



# Create both training and validation dataloaders

train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)

valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)



# Create model

model = faster_rcnn.model(num_classes=len(class_map))



# Define metrics

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = faster_rcnn.fastai.learner(

    dls=[train_dl, valid_dl], model=model, metrics=metrics

)



learn.fine_tune(2, lr=1e-4)
plt.imshow(img_np)
trained_model = learn.model
sd = trained_model.state_dict()
PATH = 'pets1.pth'

torch.save(trained_model.state_dict(), PATH)
type(trained_model)
infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])

infer_ds = Dataset.from_images([img_np], infer_tfms)

infer_dl = faster_rcnn.infer_dl(infer_ds, batch_size=1)



samples, preds = faster_rcnn.predict_dl(model=trained_model, infer_dl=infer_dl)



imgs = [sample["img"] for sample in samples]

show_preds(

    imgs=imgs,

    preds=preds,

    class_map=class_map,

    denormalize_fn=denormalize_imagenet,

    show=True,

    display_label=False,

)