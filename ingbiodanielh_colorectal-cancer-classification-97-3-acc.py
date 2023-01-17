from fastai import *

from fastai.vision import *



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve





import os

print(os.listdir("../input"))



% matplotlib inline
path= Path('../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/')

csv_file='../input/hmnist_28_28_RGB.csv'
df= pd.read_csv(csv_file)

df.head()
class_names = {1: "Tumor", 2: "Stroma", 3: "Complex", 4: "Lympho",

               5: "Debris", 6: "Mucosa", 7: "Adipose", 8: "Empty"}

class_numbers = {"Tumor": 1, "Stroma": 2, "Complex": 3, "Lympho": 4,

               "Debris": 5, "Mucosa": 6, "Adipose": 7, "Empty": 8}

class_colors = {1: "Red", 2: "Orange", 3: "Gold", 4: "Limegreen",

                5: "Mediumseagreen", 6: "Darkturquoise", 7: "Steelblue", 8: "Purple"}



label_percentage = df.label.value_counts() / df.shape[0]

class_index = [class_names[idx] for idx in label_percentage.index.values]



plt.figure(figsize=(20,5))

sns.barplot(x=class_index, y=label_percentage.values, palette="Set3");

plt.ylabel("% in data");

plt.xlabel("Target cancer class");

plt.title("How is cancer distributed in this data?");
tfms=get_transforms(flip_vert=True, max_warp=0.)
data = (ImageItemList.from_folder(path)

        .random_split_by_pct()

        .label_from_folder()

        .transform(tfms, size=150)

        .databunch(num_workers=2, bs=32))
data.show_batch(row=3, figsize=(10,10))
learner= create_cnn(data, models.resnet34, metrics=[accuracy], model_dir='/tmp/models/')
learner.lr_find()

learner.recorder.plot()
lr=5e-3

learner.fit_one_cycle(8, lr)
learner.save('stage-1')
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(16, slice(5e-5,5e-4))
learner.recorder.plot_losses()
learner.save('stage-2')
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()
preds, lb=learner.get_preds()
#  ROC curve

fpr, tpr, thresholds = roc_curve(lb.numpy(), preds.numpy()[:,1], pos_label=1)



#  ROC area

pred_score = auc(fpr, tpr)

print(f'ROC area is {pred_score}')
plt.figure()

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")