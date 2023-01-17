#Getting all the necessary imports

import numpy as np

import pandas as pd

from pathlib import Path

from fastai import *

from fastai.vision import *

import torch

import os
#Creating the data path variable and initializing the transforms

data_folder = Path("../input/flower_data/flower_data")

trfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
#creating the data loader

data = (ImageList.from_folder(data_folder)

        .split_by_folder()

        .label_from_folder()

        .add_test_folder("../input/test set/test set")

        .transform(trfms, size=128)

        .databunch(bs=64, device= torch.device('cuda:0'))

        .normalize())
#Testing the data loader

data.show_batch(3, figsize=(6,6), hide_axis=False)
#defining the learner

learn = cnn_learner(data, models.densenet161, metrics=[error_rate, accuracy], model_dir = "../../../working") #Using densenet as discussed above
#finding the learning rate

learn.lr_find(stop_div=False, num_it=200)
#plotting loss against learning rate

learn.recorder.plot(suggestion = True)

min_grad_lr = learn.recorder.min_grad_lr
#using the learning rate and starting the training

lr = min_grad_lr

learn.fit_one_cycle(60, slice(lr)) #For final model, keep number of epochs = 60
#Saving the model

learn.export(file = '../../../working/export.pkl')
#Reloading the model into the memory and using it over test data

learn = load_learner(os.getcwd(), test=ImageList.from_folder('../input/test set/test set')) #pointing the learner towards the test data
#Getting the labels from the JSON

import json

with open('../input/cat_to_name.json') as f:

  conversion_data = json.load(f)
#Creating a final list with file name, prediction category and the corresponding name

final_result = []

for i in range (len(learn.data.test_ds)):

    filename = str(learn.data.test_ds.items[i])[27:]

    pred_category  = int(learn.predict(learn.data.test_ds[i][0])[1])

    category_name = conversion_data[str(pred_category)]

    final_result.append((filename, pred_category, category_name))
#Sorting the list alphabetically

final_result = sorted(final_result,key=lambda x: x[0])
#Saving the Final Output to a CSV

final_output = pd.DataFrame(final_result, columns=["Filename", "Predicted_Category","Category_Name"])

final_output.to_csv('final_output.csv', index=False)
#Checking that the CSV is created properly

test_csv = pd.read_csv("final_output.csv")

test_csv