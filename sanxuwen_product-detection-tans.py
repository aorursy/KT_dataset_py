import os

print(os.listdir("../input"))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install git+https://github.com/fastai/fastai2 

from fastai2.vision.all import *
! pip install efficientnet-pytorch -q

from efficientnet_pytorch import EfficientNet
# train and test csv



train = pd.read_csv("../input/uploaded-dataset/train.csv")

test = pd.read_csv("../input/uploaded-dataset/test.csv")





# paths leading to images

train_path = Path("../input/uploaded-dataset/train/train")

test_path = Path("../input/uploaded-dataset/test/test")
# add the category to filename for easier usage with fastai API

train['filename'] = train.apply(lambda x: str(x.category).zfill(2) + '/' + x.filename, axis=1)

train
# train in a 10% subset of the data

# to speed up experimentation

# comment these lines out to increase accuracy (but necessitates longer training time)





# from sklearn.model_selection import train_test_split

# _, train = train_test_split(train, test_size=0.05, stratify=train.category)
#Run this to see base augmentation

# aug_transforms??
#Run this to see base normalization

# Normalize.from_stats??
item_tfms = [RandomResizedCrop(299, min_scale=0.75)] #before 224 not 456

batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]

def get_dls_from_df(df):

    df = df.copy()

    options = {

        "item_tfms": item_tfms,

        "batch_tfms": batch_tfms,

        "bs": 32,

    }

    dls = ImageDataLoaders.from_df(df, train_path, **options)

    return dls
dls = get_dls_from_df(train)

# dls.show_batch()
# pretrained models https://www.kaggle.com/mhiro2/pytorch-pretrained-models (model)

# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch (load model)

# https://github.com/Cadene/pretrained-models.pytorch#inception
import torch

import torchvision
!pip install pretrainedmodels
import pretrainedmodels

print(pretrainedmodels.model_names)
model_name = 'inceptionv4' # could be fbresnet152 or inceptionresnetv2

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# model = EfficientNet.from_pretrained("efficientnet-b5")

head = nn.Sequential(

    AdaptiveConcatPool2d(),

    Flatten(),

    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

    nn.Dropout(p=0.25),

    nn.Linear(in_features=1024, out_features=512,bias = True),

    nn.ReLU(inplace=True),

    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

    nn.Dropout(p=0.5),

    nn.Linear(in_features=512, out_features=42)

)

learn = Learner(dls, model, metrics=accuracy, custom_head = head)

learn.model_dir =  Path('..')

print(learn.model_dir)

learn = learn.load('/kaggle/input/modelssss/inceptionv4  2 epoch v2')
# base_lr = 2e-3

# learn.freeze()

# learn.fit_one_cycle(2,slice(base_lr), pct_start=0.99)



# test_images = test.filename.apply(lambda fn: test_path/fn)

# test_dl = dls.test_dl(test_images)



# preds = learn.get_preds(dl=test_dl, with_decoded=True)

# preds



# # save raw predictions

# torch.save(preds, "rawpreds")



# submission  = test[["filename"]]

# submission["category"] = preds[2]





# # zero-pad the submissions

# submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))



# # save the submissions as CSV

# submission.to_csv("submission_one_cycle.csv")





# learn.export(Path("/kaggle/working/export_one_cycle.pkl"))

# learn.model_dir='/kaggle/working/'

# learn.save('model_one_cycle')
learn.unfreeze()

learn.fit_one_cycle(2)



test_images = test.filename.apply(lambda fn: test_path/fn)

test_dl = dls.test_dl(test_images)



preds = learn.get_preds(dl=test_dl, with_decoded=True)

preds



# save raw predictions

torch.save(preds, "rawpreds")



submission  = test[["filename"]]

submission["category"] = preds[2]





# zero-pad the submissions

submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))



# save the submissions as CSV

submission.to_csv("submission_two_cycle.csv")





learn.export(Path("/kaggle/working/export_two_cycle.pkl"))

learn.model_dir='/kaggle/working/'

learn.save('model_two_cycle')
with open('inceptionv4v2_final_pred.pkl', 'wb') as f:

    pickle.dump(preds[0], f)
# learn.unfreeze()

# learn.fit_one_cycle(1)



# test_images = test.filename.apply(lambda fn: test_path/fn)

# test_dl = dls.test_dl(test_images)



# preds = learn.get_preds(dl=test_dl, with_decoded=True)

# preds



# # save raw predictions

# torch.save(preds, "rawpreds")



# submission  = test[["filename"]]

# submission["category"] = preds[2]





# # zero-pad the submissions

# submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))



# # save the submissions as CSV

# submission.to_csv("submission_one_cycle.csv")





# learn.export(Path("/kaggle/working/export_one_cycle.pkl"))

# learn.model_dir='/kaggle/working/'

# learn.save('model_one_cycle')
# learn.fit_one_cycle(1)



# test_images = test.filename.apply(lambda fn: test_path/fn)

# test_dl = dls.test_dl(test_images)



# preds = learn.get_preds(dl=test_dl, with_decoded=True)

# preds



# # save raw predictions

# torch.save(preds, "rawpreds")



# submission  = test[["filename"]]

# submission["category"] = preds[2]





# # zero-pad the submissions

# submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))



# # save the submissions as CSV

# submission.to_csv("submission_two_cycle.csv")





# learn.export(Path("/kaggle/working/export_two_cycle.pkl"))

# learn.model_dir='/kaggle/working/'

# learn.save('model_two_cycle')
# learn.fit_one_cycle(1)



# test_images = test.filename.apply(lambda fn: test_path/fn)

# test_dl = dls.test_dl(test_images)



# preds = learn.get_preds(dl=test_dl, with_decoded=True)

# preds



# # save raw predictions

# torch.save(preds, "rawpreds")



# submission  = test[["filename"]]

# submission["category"] = preds[2]





# # zero-pad the submissions

# submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))



# # save the submissions as CSV

# submission.to_csv("submission_three_cycle.csv")





# learn.export(Path("/kaggle/working/export_three_cycle.pkl"))

# learn.model_dir='/kaggle/working/'

# learn.save('model_three_cycle')
##Maybe try stratified Kfold https://www.kaggle.com/muellerzr/resnet152-with-tta-and-fine-tune-fastai2 




# learn = cnn_learner(dls, resnet34, metrics=accuracy)

# learn.fine_tune(2)

# learn.fit(4)

# learn.fit_one_cycle(4)
"""



def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,

              pct_start=0.3, div=5.0, **kwargs):

    "Fine tune with `freeze` for `freeze_epochs` then with `unfreeze` from `epochs` using discriminative LR"

    self.freeze()

    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)

    base_lr /= 2

    self.unfreeze()

    self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)

    

    

"""
# model = EfficientNet.from_pretrained("efficientnet-b5")

# num_ftrs = model._fc.in_features





# model._fc = nn.Sequential(nn.Linear(num_ftrs, 1000),

#                               nn.ReLU(),

#                              "" nn.Dropout(),

#                               nn.Linear(1000, dls.c))
# head = nn.Sequential(

#     AdaptiveConcatPool2d(),

#     Flatten(),

#     nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#     nn.Dropout(p=0.25),

#     nn.Linear(in_features=1024, out_features=512,bias = True),

#     nn.ReLU(inplace=True),

#     nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#     nn.Dropout(p=0.5),

#     nn.Linear(in_features=512, out_features=42)

# )

# learn = Learner(dls, model, metrics=accuracy, custom_head = head)

# learn.fine_tune(epochs = 2, freeze_epochs = 2)
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix(figsize=(15,15), dpi=60)
# get the most confused labels with at least 10 incorrect predictions

# interp.most_confused(10)
test_images = test.filename.apply(lambda fn: test_path/fn)

test_dl = dls.test_dl(test_images)
#https://www.kaggle.com/muellerzr/resnet152-with-tta-and-fine-tune-fastai2 fastai tta

y,_ = learn.tta(dl = test_dl)
final_predictions_tta = []

for i in range(y.shape[0]):

    final_predictions_tta.append(y[i].argmax().tolist())

final_predictions_tta
preds = learn.get_preds(dl=test_dl, with_decoded=True)

preds
count = 0

for idx,p in enumerate(final_predictions_tta):

    if p == preds[2][idx]:

        count+=1

print(count)
len(final_predictions)
# save raw predictions

torch.save(preds, "rawpreds")
submission  = test[["filename"]]

submission["category"] = preds[2]
# zero-pad the submissions

submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))
# preview

submission
# save the submissions as CSV

submission.to_csv("submission.csv")
learn.export(Path("/kaggle/working/export.pkl"))

learn.model_dir='/kaggle/working/'

learn.save('model')