# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../working"))



# Any results you write to the current directory are saved as output.

# install package from kernel, 'switch internet on' and '!pip install package_name'

## submission code - in new kernel on 'submission' page

# submission = pd.read_csv("../input/submissions/submission_1.csv")

# submission.to_csv('submission.csv', index=False)

# print(os.listdir("../input"))

# print(os.listdir("../working"))

## submission kernels

# https://www.kaggle.com/anuragtr/kernel007cee2e21/edit
# print(os.listdir("../input/working"))
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *

from fastai.metrics import *

from fastai.callbacks import *

# doc(ImageDataBunch) # to see documentation of a function
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()
train_df['diagnosis'].hist()

train_df['diagnosis'].value_counts()
def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

random_seed(120, True)
# from_csv

bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

img_size = 224

path = '../input/aptos2019-blindness-detection/'

# tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.) # customized transformations

data = ImageDataBunch.from_csv(path, folder='train_images', csv_labels='train.csv', size=img_size, suffix='.png', ds_tfms=get_transforms()).normalize(imagenet_stats)

    # ref: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb

        # https://docs.fast.ai/data_block.html

        # https://docs.fast.ai/vision.data.html
# data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
kappa = KappaScore()

kappa.weights = "quadratic"

# learn = cnn_learner(data, models.resnet34, metrics=[accuracy, kappa]) # or 'error_rate'

# learn = cnn_learner(data, models.resnet34, metrics=error_rate) # use this for only 'error_rate'

learn = cnn_learner(data, models.resnet152, metrics=[accuracy, kappa]) # or 'error_rate'

learn.model

# no layers are unfrozen, just few added at the end
# callback to save best model

# learn.fit_one_cycle(4)

learn.model_dir = '../working'

learn.fit_one_cycle(4, callbacks=[SaveModelCallback(learn, every='improvement', monitor='kappa_score', name='resnet-def-best')]) # or 'error_rate' 'accuracy'..
""" resnet34;img=224;bs=64;ep=4;seed=120                                                                       0.786885 acc; 0.835128 kappa """

""" resnet152;img=224;bs=64;ep=4;seed=120                                                                       0.788251 acc; 0.865084 kappa """
## not required when SaveModelCallback implemented

# learn.model_dir = '../working'

# learn.save('resnet34-1')

# # learn = learn.load("resnet34-1")

# print(os.listdir("../input/working"))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(9, figsize=(15,11))
# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
# continuing from default model
# find optimum lr

kappa = KappaScore()

kappa.weights = "quadratic"

learn = cnn_learner(data, models.resnet152, metrics=[accuracy, kappa])

# learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.unfreeze() # must before lr_find

learn.model_dir = '../working'

learn.load('resnet-def-best') # load default model; new training starts with weights of this model

    # Poonam's reply - https://forums.fast.ai/t/why-do-we-need-to-unfreeze-the-learner-everytime-before-retarining-even-if-learn-fit-one-cycle-works-fine-without-learn-unfreeze/41614/2

learn.lr_find()

learn.recorder.plot()

# note: if optimum lr found without loading default, it's incorrect
# architecture & training

learn.unfreeze()

# learn.fit_one_cycle(8, max_lr=slice(1e-06, 1e-04))

learn.fit_one_cycle(4, max_lr=slice(1e-06, 1e-05), callbacks=[SaveModelCallback(learn, every='improvement', monitor='kappa_score', name='resnet-def-best2')])

""" Author - "I tried to pick something well before it started getting worse. So I decided to pick 1e-6. But there's no point training 

all the layers at that rate, because we know that the later layers worked just fine before when we were training much more quickly." 

"And the first part of the slice should be a value from your learning rate finder which is well before things started getting worse. 

So you can see things are starting to get worse maybe about here(1e-4): So I picked something that's at least 10 times smaller(1e-6) than that." 

So he picked (1e-6,1e-4)    """

    # ref: https://github.com/hiromis/notes/blob/master/Lesson1.md
""" resnet34;img=224;bs=64;ep=8;lr(1e-06, 1e-04)                                                                    0.799180 acc; kappa 0.862905"""

""" resnet152;img=224;bs=64;ep=8;lr(1e-06, 1e-05)                                                                     acc; kappa """



# even after increasing epochs to 40, accuracy stays similar









# things to try:

    # unfreeze limited layers

    # https://www.kdnuggets.com/2019/05/boost-your-image-classification-model.html
## not required when SaveModelCallback implemented

# learn.model_dir = '../working'

# learn.save('resnet34-2')

# # learn = learn.load("stage-1")

# print(os.listdir("../input/working"))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix()
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))

preds,y = learn.get_preds(ds_type=DatasetType.Test)

# then implement your own logic here(highest prob is final class)/or from below notebooks

    # https://www.kaggle.com/axel81/fastai-ensembler

    # https://www.kaggle.com/demonplus/fast-ai-starter-with-resnet-50
preds
sam_sub_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sam_sub_df["id_code"]=sam_sub_df["id_code"]

print(sam_sub_df.shape)

sam_sub_df.head()
results=pd.DataFrame({"id_code":sam_sub_df["id_code"],

                      "diagnosis":np.argmax(preds,axis=1)})

results['id_code'] = results['id_code']

results['id_code'] = results['id_code'].astype('str')

results['diagnosis'] = results['diagnosis'].astype('int')

results.to_csv("submission.csv",index=False)

results.head()
""" submission """

"""  error;  Kappa   -     LB """
# Things to try:

# https://www.kdnuggets.com/2019/05/boost-your-image-classification-model.html
# np.random.seed(4)

msk = np.random.rand(len(dfx)) < 0.8

train = dfx[msk]

test = dfx[~msk]