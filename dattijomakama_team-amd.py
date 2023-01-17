import os
from fastai import *
from fastai.vision import *

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

from math import floor
os.listdir("../input/ammi-2020-convnets/train/train")
train_path = "../input/ammi-2020-convnets/train/train"
test_path = "../input/ammi-2020-convnets/test/test/0"
extra_path = "../input/ammi-2020-convnets/extraimages/extraimages"
def get_classes(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)
from glob import glob
imagePatches = glob("../input/ammi-2020-convnets/train/train/*/*.*", recursive=True)
imagePatches[0:10]

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),]) 
data = ImageDataBunch.from_name_func(path, imagePatches, label_func=get_classes,size=430, 
                                     bs=24,num_workers=2,test = test_path,ds_tfms=tfms
                                  ).normalize(imagenet_stats)
learner= cnn_learner(data, models.densenet121,metrics=[accuracy],bn_final=True,opt_func=optim.AdamW,ps = 0.25,model_dir='/tmp/models/')
learner.lr_find()
learner.recorder.plot()
lr=1e-2
learner.fit_one_cycle(1, lr)
learner.save('model-1')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.load('model-1')
learner.fit_one_cycle(8, slice(1e-5,1e-4))
learner.recorder.plot_losses()
learner.validate()
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
preds,y = learner.TTA(ds_type=DatasetType.Test)
import shutil
path=""
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),]) 
image_data_bunch = ImageDataBunch.from_name_func(path, imagePatches, label_func=get_classes,  size=500, 
                                     bs=20,num_workers=2,test = extra_path,ds_tfms=tfms
                                  ).normalize(imagenet_stats)
learner.data = image_data_bunch

# Generate the psuedo labels with the best loaded model
predicted_probs_extra, _ = learner.TTA(ds_type = DatasetType.Test)
predicted_class_probs, predicted_classes_extra = predicted_probs_extra.max(dim=1)
class_labels = np.array(['cbb','cbsd','cgm','cmd','healthy'])
predicted_class_labels = class_labels[predicted_classes_extra]
shutil.copytree("../input/ammi-2020-convnets/train/train/", "../output/kaggle/working/data/train")
shutil.copytree("../input/ammi-2020-convnets/test/test/0", "../output/kaggle/working/data/test")

threshold = 0.95  # only include pseudo-labeled images where model is sufficiently confident in its prediction
filenames = [item.name for item in learner.data.test_ds.items]
for predicted_class_label, predicted_class_probability, filename in zip(predicted_class_labels, predicted_class_probs, filenames):
#     print(predicted_class_label, predicted_class_probability)
    if predicted_class_probability > threshold:
        shutil.copy(f"../input/ammi-2020-convnets/extraimages/extraimages/{filename}", f"../output/kaggle/working/data/train/{predicted_class_label}/{filename}")
from glob import glob
imagePatches = glob("../output/kaggle/working/data/train/*/*.*", recursive=True)
imagePatches[0:10]
path = ""
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),])
test_path1 = "../output/kaggle/working/data/test"
data = ImageDataBunch.from_name_func(path, imagePatches, label_func=get_classes,  size=500, 
                                     bs=20,num_workers=2,test = test_path1,ds_tfms=tfms
                                  ).normalize(imagenet_stats)
learner= cnn_learner(image_data_bunch, models.densenet121,metrics=[accuracy],opt_func=optim.AdamW ,ps = 0.25 ,model_dir='/tmp/models')
learner.lr_find()
learner.recorder.plot()
lr=1e-2
learner.fit_one_cycle(1, lr)
learner.save('model-2')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.load('model-2')
learner.fit_one_cycle(8, slice(1e-5,1e-4))
learner.save('model-3')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.load('model-3')
lr=1e-4
learner.fit_one_cycle(1, lr)
learner.save('model-4')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.load('model-4')
learner.fit_one_cycle(15, slice(1e-5,1e-4))
learner.recorder.plot_losses()
learner.validate()
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
preds,y = learner.TTA(ds_type=DatasetType.Test)
SAMPLE_SUB = '../input/ammi-2020-convnets/sample_submission_file.csv'
sample_df = pd.read_csv(SAMPLE_SUB)
sample_df.head()
predictions = preds.numpy()

class_preds = np.argmax(predictions, axis=1)
for c, i in learner.data.train_ds.y.c2i.items():
    print(c,i)
categories = ['cbb','cbsd','cgm','cmd','healthy']

def map_to_categories(predictions):
    return(categories[predictions])

categories_preds = list(map(map_to_categories,class_preds))
filenames = list(map(os.path.basename,os.listdir(test_path)))
df_sub = pd.DataFrame({'Category':categories_preds,'Id':filenames})
df_sub.head()
# Export to csv
df_sub.to_csv('submission_categories.csv', header=True, index=False)
