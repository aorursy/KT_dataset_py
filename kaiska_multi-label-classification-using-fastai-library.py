%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *



import os

import sys

import shutil

import requests

# from PIL import Image

from io import BytesIO
# copy dataset to working (to enable manipulating the directory)

path = '/kaggle/input/apparel-dataset/'   

dest = '/kaggle/working/dataset/'

shutil.copytree(path, dest, copy_function = shutil.copy)  
os.listdir('/kaggle/working/dataset/')
tfms = get_transforms()



img_src = '/kaggle/working/dataset/'

src = (ImageList.from_folder(img_src) #set image folder

       .split_by_rand_pct(0.2) #set the split of training and validation to 80/20

       .label_from_folder(label_delim='_')) #get label names from folder and split by underscore



data = (src.transform(tfms, size=256) #set image size to 256

        .databunch(num_workers=0).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))

print(f"""Classes in our data: {data.classes}\n

Number of classes: {data.c}\n

Training Dataset Length: {len(data.train_ds)}\n

Validation Dataset Length: {len(data.valid_ds)}""")
acc_02 = partial(accuracy_thresh, thresh=0.2)

learn = cnn_learner(data, models.resnet50, metrics=acc_02, model_dir='/kaggle/working/models')
learn.fit_one_cycle(5)
learn.save('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(3e-5, 5e-4))
learn.save('stage-2-rn50')
learn.recorder.plot_losses()
# If you need to load a model, use the funciton below

# learn.load('/kaggle/input/multilabel-models/models/stage-2-rn50')
learn = load_learner('/kaggle/input/multilabel-models/models/', 

                     test=ImageList.from_folder('/kaggle/input/apparel/black_pants')) #loading from training set as an example only

preds,_ = learn.get_preds(ds_type=DatasetType.Test)
"""

Get the prediction labels and their accuracies, then return the results as a dictionary.



[obj] - tensor matrix containing the predicted accuracy given from the model

[learn] - fastai learner needed to get the labels

[thresh] - minimum accuracy threshold to returning results

"""

def get_preds(obj, learn, thresh = 15):

    labels = []

    # get list of classes from Learner object

    for item in learn.data.c2i:

        labels.append(item)



    predictions = {}

    x=0

    for item in obj:

        acc= round(item.item(), 3)*100

#         acc= int(item.item()*100) # no decimal places

        if acc > thresh:

            predictions[labels[x]] = acc

        x+=1

        

    # sorting predictions by highest accuracy

    predictions ={k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}



    return predictions
from io import BytesIO

import requests



url = "https://live.staticflickr.com/8188/28638701352_1aa058d0c6_b.jpg" 

response = requests.get(url).content #get request contents



img = open_image(BytesIO(response)) #convert to image

# img = open_image(path_to_img) #for local image file



img.show() #show image

_, _, pred_pct = learn.predict(img) #predict while ignoring first 2 array inputs

print(get_preds(pred_pct, learn))
learn.export()