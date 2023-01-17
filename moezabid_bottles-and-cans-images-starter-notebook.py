from google.colab import drive
drive.mount('/content/drive')
!ls
!mkdir Data
!unzip /content/drive/My\ Drive/Data_RVM_Vegeta.zip -d ./Data
from fastai.vision import *
path = Path('./Data/')
import numpy as np # linear algebra
import pandas as pd
tfms = get_transforms(do_flip=True,max_lighting=0.1,max_rotate=0.1)
data = (ImageDataBunch.from_folder(path,train='.',valid_pct=0.15,ds_tfms=tfms,size=224, num_workers=4)
                     .normalize(imagenet_stats))   # valid size here its 15% of total images, 
                                                # train = train folder here we use all the folder
                                                # from_folder take images from folder and labels them like wise
data.show_batch(rows=3)
len(data.classes), len(data.train_ds), len(data.valid_ds)
fb = FBeta()
fb.average = 'macro'
# We are using fbeta macro average in case some class of birds have less train images
learn = cnn_learner(data, models.resnet18, metrics=[error_rate,fb, accuracy],model_dir='./working')
learn.lr_find()
learn.recorder.plot()
lr = 1e-2 # learning rate
learn.fit_one_cycle(10,lr,moms=(0.8,0.7))  # moms
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
learn = cnn_learner(data, models.resnet101, metrics=[error_rate,fb, accuracy],model_dir='./working')
lr = 1e-2 # learning rate
learn.fit_one_cycle(10,lr,moms=(0.8,0.7))  # moms
learn = cnn_learner(data, models.alexnet, metrics=[error_rate,fb, accuracy],model_dir='./working')
lr = 1e-2 # learning rate
learn.fit_one_cycle(10,lr,moms=(0.8,0.7))  # moms