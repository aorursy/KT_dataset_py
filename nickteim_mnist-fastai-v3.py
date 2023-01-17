%reload_ext autoreload

%autoreload 2

%matplotlib inline
# FOR non-FASTAI LIBRARIES

import numpy as np 

import pandas as pd

import os

import random



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from fastai.vision import *

from fastai.metrics import *





# make sure CUDA is available and enabled

print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
path = Path("../input/digit-recognizer")

os.listdir(path)
train_df = pd.read_csv(path/"train.csv")

train_df.head()
test_df = pd.read_csv(path/"test.csv")

test_df.head()
TRAIN = Path("../train")

TEST = Path("../test")
#make directories folders for train

for i in range(10):    

    try:         

        os.makedirs(TRAIN/str(i))       

    except:

        pass



#see directory

print(os.listdir(TRAIN))
#make directories folders for test

try:

    os.makedirs(TEST)

except:

    pass



#see directory

print(os.listdir(TEST))
# os.listdir(TEST)

if os.path.isdir(TRAIN):

    print('Train directory has been created')

else:

    print('Train directory creation failed.')



if os.path.isdir(TEST):

    print('Test directory has been created')

else:

    print('Test directory creation failed.')
from PIL import Image
def pix2img(pix_data, filepath):

    img_mat = pix_data.reshape(28,28)

    img_mat = img_mat.astype(np.uint8())

    

    img_dat = Image.fromarray(img_mat)

    img_dat.save(filepath)
# save training images

for idx, data in train_df.iterrows():

    

    label, data = data[0], data[1:]

    folder = TRAIN/str(label)

    

    fname = f"{idx}.jpg"

    filepath = folder/fname

    

    img_data = data.values

    

    pix2img(img_data,filepath)
# save test images

for idx, data in test_df.iterrows():

    folder = TEST

    

    fname = f"{idx}.jpg"

    filepath = folder/fname

    

    img_data = data.values

    

    pix2img(img_data,filepath)
tfms = get_transforms(do_flip = False)
print('test : ',TEST)

print('train: ', TRAIN)

print(type(TEST))
path = ("../train")

# test = ("../test")
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", test = ("../test"), valid_pct=0.2,

        ds_tfms=get_transforms(), size=28, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(5,5))
mnist_stats
learn = cnn_learner(data, base_arch = models.resnet34, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5, 1e-03)
learn.save('model1')
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5 , 1e-04)
learn.fit_one_cycle(5 , slice(1e-05,1e-04))
learn.fit_one_cycle(5 , slice(1e-06,1e-05))
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", test = ("../test"), valid_pct=0.2,

        ds_tfms=get_transforms(), size=69, num_workers=0).normalize(imagenet_stats)



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
lr=1e-03
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-4')
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.fit_one_cycle(10, 1e-05, wd=0.5)
learn.show_results(3, figsize= (7,7))
class_score , y = learn.get_preds(DatasetType.Test)
probabilities = class_score[0].tolist()

[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
class_score = np.argmax(class_score, axis=1)
class_score[1].item()
import pandas as pd

sample_submission =  pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_submission.head()
# remove file extension from filename

ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]

# typecast to int so that file can be sorted by ImageId

ImageId = [int(path) for path in ImageId]

# +1 because index starts at 1 in the submission file

ImageId = [ID+1 for ID in ImageId]
submission  = pd.DataFrame({

    "ImageId": ImageId,

    "Label": class_score

})



submission.to_csv("submission.csv", index=False)
submission.head()