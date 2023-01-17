import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

path = "../input/histopathologic-cancer-detection/"

data = pd.read_csv(path +"train_labels.csv")
train_path = path +'train/'
test_path = path + 'test/'
# quick look at the label stats
data['label'].value_counts()
def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img
# random sampling
shuffled_data = shuffle(data)

fig, ax = plt.subplots(2,5, figsize=(20,8))
fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[0,i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')
    ax[0,i].add_patch(box)
ax[0,0].set_ylabel('Negative samples', size='large')
# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')
    ax[1,i].add_patch(box)
ax[1,0].set_ylabel('Tumor tissue samples', size='large')
data.head()
!pip install -Uqq fastbook
import fastbook
from fastai.vision.all import *
path = "../input/histopathologic-cancer-detection/"

def get_x(r): return path+'train/'+r['id']+'.tif'
def get_y(r): return r['label']


# start with creatinga datablock

dblock =  DataBlock(blocks=(ImageBlock, CategoryBlock),
                    splitter=RandomSplitter(valid_pct=0.2,seed=42), 
                    get_x=get_x, 
                    get_y=get_y, 
                    item_tfms=RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(data)


dls.show_batch(nrows=1, ncols=3)
dblock =  DataBlock(blocks=(ImageBlock, CategoryBlock),
                    splitter=RandomSplitter(valid_pct=0.2,seed=42), 
                    get_x=get_x, 
                    get_y=get_y, 
                    item_tfms= (CropPad(48, pad_mode='zeros'),DihedralItem(p=1.0, nm=None, before_call=None) ))
dls = dblock.dataloaders(data)
dls.show_batch()
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(16, nrows=8)
learn.predict('../input/histopathologic-cancer-detection/test/00006537328c33e284c973d7b39d340809f7271b.tif')
path = "../input/histopathologic-cancer-detection/"
preds, y = learn.tta()
acc = accuracy(preds, y)

from sklearn.metrics import roc_auc_score
def auc_score(y_pred,y_true,tens=True):
    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score = tensor(score)
    return score
print('The validation accuracy is {} %.'.format(acc * 100))
pred_score = auc_score(preds,y).item()
print('The validation AUC is {}.'.format(pred_score))
# # doesnt work
# tf_fns = get_image_files(path + 'test')
# test_data = DataBlock(get_items=get_image_files,
#                  item_tfms=(CropPad(48, pad_mode='zeros'),DihedralItem(p=1.0, nm=None, before_call=None)))
# dl_test = test_data.dataloaders(path+'test')
# dl_test.show_batch()


test_images = get_image_files(path + 'test')
preds,y = learn.get_preds(dl=dls.test_dl(test_images, shuffle=False, drop_last=False))
pred_list = list(preds[:,1])
len(pred_list), len(test_images)
submissions = pd.read_csv(path + 'sample_submission.csv')
id_list = list(submissions.id)
id_list
test_images_dict = {}
for i in range(len(test_images)):
    test_images_dict[str(str(test_images[i]).split('/')[-1].split('.')[0])] = float(pred_list[i])
test_images_dict['88a12685148c0d876fed1fba8228afc6e7ee937f']
prediction_list  = []

for i in id_list:
    prediction_list.append(test_images_dict[i])
prediction_list[:5]

submissions = pd.DataFrame({'id':id_list,'label':prediction_list})
submissions.to_csv("submission.csv".format(pred_score),index = False)
from fastai.vision.widgets import *
btn_upload = widgets.FileUpload()
btn_upload
img =   PILImage.create(btn_upload.data[-1])
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(600,600))
out_pl