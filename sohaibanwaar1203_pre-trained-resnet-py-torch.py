# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

from sklearn.utils import shuffle

# Making pretrained weights work without needing to find the default filename

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'



%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv("../input/train.csv")

df.head()
from PIL import Image 

from skimage.transform import resize

train=pd.read_csv("../input/train.csv")

train_images=[]

path="../input/train/train/"

for i in train.id:

    image=plt.imread(path+i)

    train_images.append(image)
train_images=np.asarray(train_images)

X=train_images

y=train.has_cactus

print("Labels: ",y.shape)

print("images: ",X.shape)
from keras.utils import np_utils

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

Cat_test_y = np_utils.to_categorical(y_test)

y_train=np_utils.to_categorical(y_train)



print("X_train shape : ",X_train.shape)

print("y_train shape : ",y_train.shape)

print("X_test shape : ",X_test.shape)

print("y_test shape : ",y_test.shape)
bs = 64 #smaller batch size is better for training, but may take longer

sz=32
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)

src = (ImageList.from_df(df=df,path="../input/train/train/",cols='id') #get dataset from dataset

        .split_by_rand_pct(0.2) #Splitting the dataset

        .label_from_df(cols='has_cactus') #obtain labels from the level column

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation

        .databunch(bs=bs,num_workers=4) #DataBunch

        .normalize(imagenet_stats) )#Normalize     
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
print(data.classes)

len(data.classes),data.c
data.show_batch(rows=3, figsize=(7,6))

import torchvision

from fastai.metrics import *

from fastai.callbacks import *

# build model (use resnet34)

learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir=".")
learn.fit_one_cycle(40, max_lr=slice(1e-6,4e-2))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
sample_df = pd.read_csv('../input/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,'../input/test',folder='test/'))
preds,y = learn.get_preds(ds_type=DatasetType.Test)
sample_df.has_cactus = preds.argmax(dim=-1).numpy().astype(int)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)
preds.argmax(dim=-1).numpy().astype(int)