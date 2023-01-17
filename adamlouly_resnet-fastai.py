# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import *
def folder_name(number):

    if(len(str(number))==6):

        return number

    gap = 6 - len(str(number))

    return gap *'0' + str(number)

def from_preds_to_list(preds):

    p=to_np(preds)

    lista =[]

    for i in range(len(p)):

        lista.append(np.where(np.amax(p[i])==p[i])[0][0])

    last=[]

    for element in lista:

        last.append(data.classes[element])

    return last
path="/kaggle/input/vehicle/train/train/"

np.random.seed(42)

data = ImageDataBunch.from_folder(path+'.', train=path+'.', valid_pct=0.2,

                                  ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

len(data.classes)
data.show_batch(rows=3, figsize=(7, 8))
from fastai.metrics import error_rate # 1 - accuracy

learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir="/tmp/model/")
defaults.device = torch.device('cuda') # makes sure the gpu is used

learn.fit_one_cycle(1)
learn.model_dir='/kaggle/working/'

learn.export("/kaggle/working/export50.pkl")
#learn = load_learner("../input/kerneldfa3fd74eb/")
#load_test_data

submission = pd.read_csv("/kaggle/input/vehicle/sample_submission.csv")

testpath="/kaggle/input/vehicle/test/"

submission1 = submission

submission1['Id'] = submission['Id'].apply(lambda x: folder_name(x))

submission1

learn.data.add_test(ImageList.from_df(

    submission1, testpath,

    folder='testset',

    suffix='.jpg'

))
preds, _ = learn.get_preds(DatasetType.Test)
submission["Category"]=from_preds_to_list(preds)
submission.to_csv("submit2.csv",index=False)