%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from sklearn.model_selection import train_test_split

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#delete pictures in folder

# shutil.rmtree('./data/composites/')

train = pd.read_json('../input/sar-iceberg/train.json')

test = pd.read_json('../input/sar-iceberg/test.json')

train.head()
sns.countplot(train.is_iceberg).set_title('Target variable distribution')
# get more info on dataset

train.info(), test.info()
#will check to repositioning picture according to angle of satellite.

train.inc_angle = train.inc_angle.replace('na', 0, inplace=True)

train.inc_angle.describe()
#let's see an image

img1 = train.loc[5, ['band_1', 'band_2']]

img1 = np.stack([img1['band_1'], img1['band_2']], -1).reshape(75, 75, 2)

plt.imshow(img1[:, :, 0] )
plt.imshow(img1[:, :, 1] )
#shape of dataset

train.shape
#this will help us to have more images and allowing us to see more charactersitics in image

def color_composite(data):

    rgb_arrays = []

    for i, row in data.iterrows():

        band_1 = np.array(row['band_1']).reshape(75, 75) 

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2



        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))

        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))

        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        

        rgb = np.dstack((r, g, b))

        rgb_arrays.append(rgb)

    return np.array(rgb_arrays)
rgb_train = color_composite(train)

rgb_test = color_composite(test)
print('The train shape {}'.format(rgb_train.shape))

print('The test shape {}'.format(rgb_test.shape))
#look at some ships

ships = np.random.choice(np.where(train.is_iceberg ==0)[0], 9)

fig = plt.figure(1,figsize=(12,12))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = rgb_train[ships[i], :, :]

    ax.imshow(arr)

    

plt.show()
#look at some iceberg

iceberg = np.random.choice(np.where(train.is_iceberg ==1)[0], 9)

fig = plt.figure(1,figsize=(12,12))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = rgb_train[iceberg[i], :, :]

    ax.imshow(arr)

    

plt.show()
#for creating labels and folders

#we have to label if iceberg or ship therefore I will create 2 folders - one ship, one iceberg in train set.

iceberg = train[train.is_iceberg==1]

ship = train[train.is_iceberg==0]
#save images to disk

os.makedirs('./data/composites', exist_ok= True)

os.makedirs('./data/composites/train/ship', exist_ok=True)

os.makedirs('./data/composites/train/iceberg', exist_ok=True)

os.makedirs('./data/composites/test', exist_ok=True)



#save train iceberg images

for idx in iceberg.index:

    img = rgb_train[idx]    

    plt.imsave('./data/composites/train/iceberg_' + str(idx) + '.png',  img)

        

#save train ship images

for idx in ship.index:

    img = rgb_train[idx]    

    plt.imsave('./data/composites/train/ship_' + str(idx) + '.png',  img)



       

#save test images

for idx in range(len(test)):

    img = rgb_test[idx]

    plt.imsave('./data/composites/test/' + str(idx) + '.png',  img)
# GPU required

torch.cuda.is_available()
torch.backends.cudnn.enabled
#copy model to kernel resnet 34

# Fix to enable Resnet to live on Kaggle - creates a writable location for the models

cache_dir = os.path.expanduser(os.path.join('~', '.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

   # print("directory created :" .cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

  #  print("directory created :" . cache_dir)
!cp ../input/resnet34/resnet34.pth ~/.torch/models/resnet34-333f7ec4.pth 
#get pictures/files directory

path = '../working/data/composites/'

path_img = '../working/data/composites/train/'

fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.png$'

                     

tfms = get_transforms(do_flip=True, flip_vert=True, max_lighting=0.3, max_warp=0.2)

data = ImageDataBunch.from_name_re(path_img, fnames, pat, valid_pct=0.3 ,  ds_tfms=tfms , size=128, bs= 256 , resize_method=ResizeMethod.CROP, 

                     padding_mode='reflection').normalize(imagenet_stats)  #imagenet stats

#convert image to grayscale

for itemList in ["train_dl", "valid_dl", "fix_dl", "test_dl"]:

    itemList = getattr(data, itemList)

    if itemList: itemList.x.convert_mode = "L"

data.classes
#let's check image + label

data.show_batch(rows=3,figsize=(7,8))
#create a learner

learn = cnn_learner(data, models.resnet34, metrics=[error_rate,accuracy] ,model_dir= '/tmp/models/')
learn.summary()
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9,figsize =(9,9))
interp.plot_confusion_matrix()
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-01

learn.fit_one_cycle(5,1e-02)
learn.save('stage-2')
learn.recorder.plot_losses()
#Initiating refit and checking LR

learn.unfreeze

learn.lr_find()

learn.recorder.plot(suggestion=True)
# access the corresponding learning rate 

# min_grad_lr = learn.recorder.min_grad_lr

# min_grad_lr
learn.fit_one_cycle(10, slice(1e-02, 1e-01))
learn.save('stage-3')
learn.recorder.plot_losses()
#create a new learner

# learn = cnn_learner(data, models.resnet34, metrics=[error_rate,accuracy] ,callback_fns=[partial(SaveModelCallback)],

#                     wd=0.1,ps=[0.9, 0.6, 0.4])

# learn = learn.load('stage-3')
unfrozen_validation = learn.validate()

print("Final model validation loss: {0}".format(unfrozen_validation[0]))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(2,2))
test = pd.read_json('../input/sar-iceberg/test.json')

Xtest = get_images(test)

test_predictions = model.predict_proba(Xtest)

submission = pd.DataFrame({'id': test['id'], 'is_iceberg': test_predictions[:, 1]})

submission.to_csv('sub_submission.csv', index=False)