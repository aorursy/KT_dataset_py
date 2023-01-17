# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

from fastai.vision import *

from fastai.metrics import error_rate
torch.cuda.set_device(0)
torch.cuda.get_device_name()
traindf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

traindf.head()
traindf.shape
classdata = (traindf.healthy + traindf.multiple_diseases+

             traindf.rust + traindf.scab)
classdata.head()
any(classdata > 1)
#pathstr = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

traindf["image_id"] =traindf["image_id"].astype("str") + ".jpg"

traindf.head()
traindf["label"] = (0*traindf.healthy + 1*traindf.multiple_diseases+

             2*traindf.rust + 3*traindf.scab)

traindf.drop(columns=["healthy","multiple_diseases","rust","scab"],inplace=True)
traindf.head()
# Creation of transformation object

transformations = get_transforms(do_flip = True,

                                 flip_vert=True, 

                                 max_lighting=0.1, 

                                 max_zoom=1.05,

                                 max_warp=0.,

                                 max_rotate=15,

                                 p_affine=0.75,

                                 p_lighting=0.75

                                )
pathofdata = "/kaggle/input/plant-pathology-2020-fgvc7/"
data  = ImageDataBunch.from_df(path=pathofdata, 

                               df=traindf, 

                               folder="images",

                               label_delim=None,

                               valid_pct=0.2,

                               seed=100,

                               fn_col=0, 

                               label_col=1, 

                               suffix='',

                               ds_tfms=transformations, 

                               size=512,

                               bs=64, 

                               val_bs=32,

                               )
data.show_batch(rows=3, figsize=(10,7))
data = data.normalize()
learner = cnn_learner(data, 

                      models.resnet34, 

                      pretrained=True

                      ,metrics=[error_rate, accuracy]).to_fp16()
learner.model_dir = '/kaggle/working/models'
learner.lr_find(start_lr=1e-07,end_lr=0.2, num_it=100) 

learner.recorder.plot(suggestion=True)
mingradlr = learner.recorder.min_grad_lr

print(mingradlr)
lr = mingradlr

learner.fit_one_cycle(10, lr)

learner.unfreeze()

learner.lr_find(start_lr=1e-07,end_lr=0.2, num_it=100) 

learner.recorder.plot(suggestion=True)
mingradlr1 = learner.recorder.min_grad_lr

print(mingradlr1)
# Differential learning 

learner.fit_one_cycle(7, slice(mingradlr1, mingradlr1/20))
learner.show_results()
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix(title='Confusion matrix')
testdf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

testdf.head()
sampsubmit = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")

sampsubmit.head()
pathofdata = "/kaggle/input/plant-pathology-2020-fgvc7/"
testdata= ImageList.from_folder(pathofdata+"images")
testdata.filter_by_func(lambda x: x.name.startswith("Test"))
testdata.items[0]
img1 = open_image(testdata.items[0])

img2 = open_image(testdata.items[1])
learner.predict(img1)
val1 = learner.predict(img1)[2].tolist()

val2 = learner.predict(img2)[2].tolist()
val1
tdtd = testdata.items[0]
tdtd.name[:-4:]
resultlist = []

for item in testdata.items:

    img = open_image(item)

    predval = learner.predict(img)[2].tolist()

    predval.insert(0,item.name[:-4:])

    resultlist.append(predval)
resultlist[0:5]
resultdf = pd.DataFrame(resultlist)

resultdf.columns = sampsubmit.columns

resultdf.head()
resultdf.set_index("image_id",inplace=True)

resultdf.head()
resultdf = resultdf.loc[sampsubmit.image_id,:]

resultdf.head()
resultdf.reset_index(inplace=True)
resultdf.head()
resultdf.to_csv("submit.csv",index=False)