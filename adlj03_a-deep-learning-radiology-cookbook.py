# This Python 3 environment is this kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebrac

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

import csv

import glob

import os



currPath = os.getcwd()

os.mkdir(currPath+"/data")

os.mkdir(currPath+"/data/positive")

os.mkdir(currPath+"/data/negative")



print('Great! You clicked on it correctly.')
finding = "atelectasis"
# https://www.kaggle.com/rtatman/import-functions-from-kaggle-script [todo]

# search csv for finding, and return arrays of positive and negative cases

def lookfor(finding):

    reader = csv.reader(open("../input/sample/sample_labels.csv"), delimiter=",")

    positive, negative = [], []

    PREFIX = "../input/sample/sample/images/"

    next(reader) # skip header line

    for row in reader:

        if finding.lower() in row[1].lower(): positive.append(PREFIX+row[0]) # row[0] is filename

        else: negative.append(PREFIX+row[0])

    print("positive: "+str(len(positive))+" images")

    print("negative: "+str(len(negative))+" images")

    size = min(len(positive), len(negative)) # take the smaller of two numbers

    print()

    if size > 100: print("Great! We'll use "+str(size)+" positive and "+str(size)+" negative cases in our set.")

    else: print("We don't have enough data to work with. Let's try another finding.")

    return positive, negative, size



# use arrays of filenames to rearrange positive and negative cases into respective folders

def moveimages(positive, negative, size):

    # wipe directories first

    for file in glob.glob('/kaggle/working/data/positive/*.png'):

        os.unlink(file)

    for file in glob.glob('/kaggle/working/data/negative/*.png'):

        os.unlink(file)

    for x in range(size):

        shutil.copy(positive[x], "/kaggle/working/data/positive")

        shutil.copy(negative[x], "/kaggle/working/data/negative")
positive, negative, size = lookfor(finding)
path = Path('/kaggle/working/data/')

moveimages(positive, negative, size)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=256, num_workers=4).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(16,16))
learn = create_cnn(data, models.resnet34, metrics=[accuracy, AUROC()])

# this line is necessary to save the model (usually read-only access)

learn.model_dir='/kaggle/working/'
learn.lr_find(start_lr=1e-5, end_lr=5e-1)

learn.recorder.plot()
learn.fit_one_cycle(25, max_lr=slice(5e-4,5e-3))

learn.recorder.plot_losses()

learn.save('stage-1')
learn.load('stage-1');

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(16, figsize=(24,24), heatmap=False)

interp.plot_top_losses(16, figsize=(24,24), heatmap=True, alpha=0.3)