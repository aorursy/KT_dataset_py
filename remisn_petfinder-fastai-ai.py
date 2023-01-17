%reload_ext autoreload

%autoreload 2

%matplotlib inline

import numpy as np

import pandas as pd
import fastai

print("fastai: ",fastai.__version__)

import torch

print("Torch: ",torch.__version__)

import torchvision

print("Torchvision: ",torchvision.__version__)

import sklearn

print("sklearn: ",sklearn.__version__)

import sys

print("Python: ",sys.version)
# if torch.cuda.is_available():

#     devID=torch.cuda.current_device()

#     print("GPU: ",torch.cuda.get_device_name(devID))

# else:

#     print("Torch Cuda not avaialbe")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device
from fastai.vision import *

from fastai.metrics import error_rate

from fastai import *
PATH = '../input/petfinder-adoption-prediction/'

OUT_PATH = './'
import os

print(os.listdir(OUT_PATH))
import os

print(os.listdir(PATH))
!pwd
# !ls
import os

print(os.listdir(PATH+'test'))
trainCSV = pd.read_csv(PATH+'train/train.csv')

trainCSV.head().T
trainID = trainCSV[['PetID', 'AdoptionSpeed']].copy()

trainID.head()
# trainID.info()
# trainID.describe()
classes = trainID['AdoptionSpeed'].value_counts()

classes
# classes.shape, type(classes)
#check if target has any missing values

trainID['AdoptionSpeed'].isnull().values.any()
testCSV = pd.read_csv(PATH+'test/test.csv')

testCSV.head().T
testID = testCSV[['PetID']].copy()

testID.head()
path_img = PATH + 'train_images'

fnames = get_image_files(path_img)

fnames[:5]
# open_image(fnames[0])
# # print few images from train set

# fig=plt.figure(figsize=(20, 15))

# columns = 4

# rows = 4

# for i in range(1, columns*rows +1):

#     img = plt.imread(fnames[i])

#     fig.add_subplot(rows, columns, i)

#     plt.imshow(img)

# plt.show()
# # read image size

# im=array([list(open_image(image).size) for image in fnames])

# type(im)
# histogram of x

# plt.hist(im[:,0]);
# histogram of y

# plt.hist(im[:,1]);
tfms = get_transforms(do_flip=True)
#view image augmentations

def get_ex(): return open_image(fnames[0])



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

     rows,cols,figsize=(width,height))[1].flatten())]
# plots_f(3, 4, 12, 6, size=224)
trainID.head()
#pattern to parse full file path+name

# group(0) - file name with index and jpg extension

# group(1) - file name with index but without jpg extension

# group(2) - file name without index nor jpeg extension

# group(3) - file name index number

pat = r"(([\w\.-]+)-(\d+))\.jpg"

pat = re.compile(pat)

pat
#test and verify re parsing pattern

res = pat.search("../input/train_images/8e0d65b3e-1.jpg")

res.group(0), res.group(1), res.group(2), res.group(3)
#create new empty DataFrame for each File image in the row and copied Adption Speed value

NewList =  pd.DataFrame(columns=["PetID","AdoptionSpeed"])

NewList
# Loop through every file name, find match in Train Target DataFrame, extract Adoption Speed value, and 

# append new row into NewList DataFrame

for name in fnames:

    #parse file path+name

    res = pat.search(str(name))

    #print("core name: ", res.group(2))      #file_core = res.group(2)

    if ((trainID['PetID']==res.group(2)).values.any()): #if fname core is found in dataframe with PetID,AdoptionTime

            AdoptionSpeed=trainID[trainID['PetID']==res.group(2)].AdoptionSpeed.values[0] #extract Adoption Time

            #copy row into new Data Frame

            NewList = NewList.append(pd.DataFrame({"PetID":[res.group(1)], "AdoptionSpeed":[AdoptionSpeed] }),ignore_index = True)
NewList.head()
data = ImageDataBunch.from_df(PATH, NewList, folder='train_images', ds_tfms=tfms, size=224, suffix='.jpg', bs=8)
#Number of files

len(fnames)
data.normalize(imagenet_stats)
#show few images from data set with Adoptoin Speed as class above

# data.show_batch(rows=2, figsize=(10,8), ds_type=DatasetType.Train)
#list data classes

print(data.classes)

len(data.classes)
#doc(create_cnn)
kappa = KappaScore()

kappa.weights = "quadratic"

print(os.listdir("../input/"))
Path('/tmp/.cache/torch/checkpoints/').mkdir(parents=True, exist_ok = True)

shutil.copy('../input/resnet34fastai/resnet34.pth', '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')
#learn = cnn_learner(data, models.resnet34, path = OUT_PATH, metrics=error_rate)

learn = cnn_learner(data, models.resnet34, path = OUT_PATH, metrics=[kappa, accuracy])
learn.model
learn.fit_one_cycle(3)
learn.recorder.plot_losses()
learn.save('224_pre')
learn.load('224_pre');
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
#learn.fit_one_cycle(3, max_lr=slice(1e-3,5e-2))
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-5))
learn.recorder.plot_losses()
#learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-5))
learn.save('224_all')
learn.load('224_all');
# interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()

# len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(9, figsize=(15,11))
# # Run second itteration with size = 299

# data = ImageDataBunch.from_df(PATH, NewList, folder='train_images', ds_tfms=tfms, size=299, suffix='.jpg', bs=8)

# data.normalize(imagenet_stats)
# learn.data = data

# data.train_ds[0][0].shape
# # preload last weights

# learn.load('224_all')
# learn.freeze()

# learn.lr_find(start_lr=1e-8)

# learn.recorder.plot()
# learn.fit_one_cycle(3, max_lr=1e-4)
# learn.recorder.plot_losses()
# learn.save('299_pre')
# learn.load('299_pre')
# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(3, max_lr=slice(1e-7,2e-6))
# learn.recorder.plot_losses()
# learn.save('299_all')
learn.export()

learn.path
# testing DataFrame

test = ImageList.from_folder(PATH+'test_images')

len(test)
learn = load_learner(path=learn.path, test=test)
learn.data.test_ds.items
learn.data.test_ds.items[5]
learn.data.test_dl.device
learn.data.test_ds
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# find highest probability 

Prediction = preds.argmax(dim=1)

Prediction = Prediction.numpy().tolist()

Prediction[:10]
# Prediction.shape
submission = pd.read_csv(PATH+'test/sample_submission.csv')

submission.head()
submission.PetID[0]
import statistics

import math
# Extract file names wthout .jpg extension and path

fnamesShort = [f.name[:-4] for f in learn.data.test_ds.items]

fnamesShort[:10]
TotalPhotoMissing = 0

# Iterate DataFrame for each index and row (index and value)

for index, PetID in submission.iterrows():

    #print(index, PetID.values[0])  #extract Submission index and PetID core

    

    #find indices of all files in fnames that match PetID core name

    indices = [i for i, s in enumerate(fnamesShort) if PetID.values[0] in s]  

    #print(indices)

    PetIdPredictions = [Prediction[i] for i in indices] # get list of predictions with given indeces

    #print(PetIdPredictions)

    if( len(PetIdPredictions) == 0):

        TotalPhotoMissing += 1

        #print("? ",index, indices, PetID.values[0])

        FinalPrediction=2 # set to a common class

    else:

        FinalPrediction=math.ceil(statistics.median([Prediction[i] for i in indices])) #median with rounding up

    #print(FinalPrediction)

    submission.AdoptionSpeed[index] = FinalPrediction

print (" Test set has ",TotalPhotoMissing, "missing images" )
submission.head()
submission.to_csv('submission.csv', index=False)