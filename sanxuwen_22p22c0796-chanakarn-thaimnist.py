# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install fastai
from fastai.data.all import *

from fastai.vision.core import *

from fastai.vision.data import *

from fastai.vision.augment import *

from fastai.vision.learner import *
train = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')

test = pd.read_csv('../input/thai-mnist-classification/submit.csv')



train.shape, test.shape
train_path = "../input/thai-mnist-classification/train"

test_path = "../input/thai-mnist-classification/test"
item_tfms = [RandomResizedCrop(224, min_scale=0.75)]

batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]

def get_dls_from_df(df):

    df = df.copy()

    options = {

        "item_tfms": item_tfms,

        "batch_tfms": batch_tfms,

        "bs": 4,

    }

    dls = ImageDataLoaders.from_df(df, train_path, **options)

    return dls
dls = get_dls_from_df(train)

dls.show_batch()
! pip install efficientnet-pytorch -q
from efficientnet_pytorch import EfficientNet
## https://www.kaggle.com/sanxuwen/monkey-recognition-fastai2-efficientnet



model = EfficientNet.from_pretrained("efficientnet-b7")

num_ftrs = model._fc.in_features



# Replace the last fully connected layer with our own layers, I add in an additional Dropout layer for some regularizing effects

model._fc = nn.Sequential(nn.Linear(num_ftrs, 1000),

                              nn.ReLU(),

                              nn.Dropout(),

                              nn.Linear(1000, dls.c))
# ! pip install fastai2 -q

from fastai.vision.all import *

opt_func = partial(Adam)

loss_func = LabelSmoothingCrossEntropy()

metrics = [accuracy]
learn = Learner(dls, model, opt_func=opt_func, loss_func=loss_func, metrics=metrics).to_fp16()

learn.fine_tune(5)
# learn.export('../output/kaggle/working')
final_train = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
final_train.head(5)
final_train.isna().mean()
pd.isna(final_train['feature1'][0])
test_path
files = get_image_files(test_path)
file_list = []

pred_list = []

for file in files:

    pred = learn.predict(file)[2]

    pred = np.argmax(pred).item()

    pred_list.append(pred)

    file_list.append(file) 

final_pred = pd.DataFrame({"id":file_list,'category':pred_list})
final_pred['id']
final_pred['id'] = final_pred['id'].astype(str).str[40:]
### This one is test set

final_pred
files = get_image_files(train_path)
file_list = []

pred_list = []

for file in files:

    pred = learn.predict(file)[2]

    pred = np.argmax(pred).item()

    pred_list.append(pred)

    file_list.append(file) 

final_pred_train = pd.DataFrame({"id":file_list,'category':pred_list})
final_pred_train['id'] = final_pred_train['id'].astype(str).str[41:]
final_pred_train
def get_predictions(img):

    if img in prediction_dict:

        prediction = prediction_dict[img]

        return prediction

    else:

        return 'No Prediction'
# Do as key value

final_pred
# feature1 = []

# feature2 = []

# feature3 = []

# for index, row in final_train.iterrows():

#     path = '../input/thai'

#     if pd.isna(row['feature1']) == True:

#         feature1.append('No Data')

#     else:

        

#     print(row['feature1'], row['feature2'])
from fastai.vision.all import *

learn = cnn_learner(dls, resnet152,metrics=accuracy)

learn.fine_tune(3)  #resnet152 about 97
xsenet154
dlsw