#Initialize packages.

import pandas as pd
import numpy as np
import requests
import shutil
import os.path
from os import path
default = pd.read_csv('../input/coins-dataset/source_csv/coinlist.csv',encoding='ISO-8859-1')
df = default[['id','broadperiod', 'imagedir', 'filename']]
df.head(5)
def download_img(broadperiod, imagedir, filename, idnum):
    # Sets the url and filename for the image. (Need to append the period name to filename for ML processing later.)
    # The parameters are converted to string objects in case they are initially numeric.
    image_url = "https://finds.org.uk/" + str(imagedir) + "medium/" + str(filename)
    filename = str(broadperiod) + '__' + str(idnum) + '.jpg'
    
    # Check if the file already exists before running requests.get. (Useful for rerunning batch_dl.)
    if path.exists('data/images/' + filename):
        print(filename + " already exists.")
    
    # When the file doesn't exist, downloads the file.
    else:
        r = requests.get(image_url, stream=True)
    
        # If the file is accessible via r, download it.
        if r.status_code == 200:
            r.raw.decode_content = True
        
            with open('data/images/' + filename, 'wb') as f:
                shutil.copyfileobj(r.raw,f)
        
            print('Downloaded: ', str(filename))
        else:
            print("Image couldn't be retreived")

            
    # This function iterates over a dataframe.
def batch_dl(df):
    i = 0
    while i < len(df):
        download_img(df.broadperiod[i], df.imagedir[i], df.filename[i], df.id[i])
        i += 1
gr = pd.read_csv('../input/coins-dataset/source_csv/greek_provincal.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
med = pd.read_csv('../input/coins-dataset/source_csv/medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
pmed = pd.read_csv('../input/coins-dataset/source_csv/post_medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
emed = pd.read_csv('../input/coins-dataset/source_csv/early_medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]

big = pd.concat([df, gr, med, pmed, emed]).drop_duplicates()
big.head()
# If you want to run this notebook off Kaggle, you can uncomment this to download the images.
# batch_dl(big)
# Import fastai modules
from fastai.vision import *
from fastai.metrics import error_rate
# Get the file list for our image dataset.
fnames = get_image_files('../input/coins-dataset/images')
# Print part of the list.
fnames[:11]
# Set up regex for finding image filenames to find labels for image names.

pat = r'/([^/m]+?(?=__))'
imagepath = '../input/coins-dataset/images'
data = ImageDataBunch.from_name_re(imagepath, fnames, pat, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/kaggle/working/")
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes),data.c
data = ImageDataBunch.from_name_re(imagepath, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=32).normalize(imagenet_stats)
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1-50')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-2))
learn.save('stage-2-50')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12))
interp.plot_top_losses(9, figsize=(15,15))
learn.export('/kaggle/working/export.pkl')
defaults.device = torch.device('cpu')
img = open_image('../input/coins-dataset/images/EARLY MEDIEVAL__1000545.jpg')
img
learn = load_learner('/kaggle/working')
pred_class,pred_idx,outputs = learn.predict(img)
pred_class.obj