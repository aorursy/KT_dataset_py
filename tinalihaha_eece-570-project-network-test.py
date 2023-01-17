# import libraries

from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torchvision

from fastai.metrics import *

from fastai.callbacks import *
# reading CSV file containing imae labels

df = pd.read_csv('../input/homofiltlabel/homo_filt_label.csv')



# creating dataframe for training and adding images path to the label CSV

df['path'] = df['image'].map(lambda x: os.path.join('../input/clahe-reduced/clahe/','{}.jpeg'.format(x)))

df = df.drop(columns=['image'])



# Shuffle dataframe

df = df.sample(frac=1).reset_index(drop=True) 



# seperating labels into 0 as no disease and >=1 as diseased

df['level'] = (df['level'] > 1).astype(int)



# checking the dataset distrbution 

df['level'].hist(figsize = (10, 5))
# performing trainset and validation set split 80/20 similar distribution of different claases in validation set

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df,test_size=0.2) 

df = pd.concat([train_df,val_df]) #beginning of this dataframe is the training set, end is the validation set
# Since the dataset is highly imbalance with more no disease class, data augmenter is defined inclduing rotation, zooming, flipping

tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.0)

# loading dataset into imageitemlist by fastai

src = (ImageList.from_df(df=df,path='./',cols='path') #get dataframe from dataset

       .split_by_idx(range(len(train_df)-1,len(df))) #Splitting the dataset

        .label_from_df(cols='level') #obtain labels from the level column

      )

# performing data augmentation, define data bunch and normalize 

data= (src.transform(tfms,size=512,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 

        .databunch(bs=16,num_workers=4) 

        .normalize(imagenet_stats)  

       )
# import cnn learner with pretained renet 50 implemented by fastai

learn = cnn_learner(data, models.resnet50, wd = 1e-5, metrics = [accuracy,AUROC()],callback_fns=[partial(CSVLogger,append=True)])
# leaning rate finder implemented by fastai

learn.lr_find()
learn.recorder.plot()
# Fitting top classifier with the learning rate corresponding to fastest decrease of the loss

learn.fit_one_cycle(1,max_lr = 1e-2)
# plotting trainning and validation loss with number of batch processed

learn.recorder.plot_losses()
# save the first model with trained head

learn.save('model-1')
# loead previous model

learn.load('model-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(16, max_lr=slice(1e-6,1e-1))
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.save('model-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
# try predicting in a single image 

img = learn.data.train_ds[0][0]

learn.predict(img)
# predicting validation set

from sklearn.metrics import confusion_matrix



probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 

tn, fp, fn, tp = confusion_matrix(val_labels,(probs[:,1]>0.5).float()).ravel()



specificity = tn/(tn+fp)

sensitivity = tp/(tp+fn)



print('Specificity and Sensitivity',specificity,sensitivity)
# export trained model

learn.export()