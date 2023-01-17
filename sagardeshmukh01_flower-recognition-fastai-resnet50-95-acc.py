# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from fastai.vision import *



import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
#Check the folder where images are kept

path = Path('/kaggle/input/flowers-recognition/flowers/flowers/')

path.ls()
#Set the 5 output classes

classes = ['sunflower','tulip','rose','dandelion','daisy']
#Preparing the data bunch object which holds the image data

#define the batch size

bs = 16



#lets do some data augmentation

#tfms = get_transforms(do_flip=False)



data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, bs=bs, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))
#lets recheck the classes

data.classes
#lets check the stats

data.classes, data.c, len(data.train_ds), len(data.valid_ds)
#define the model 

#Make sure you have the internet switch in kaggle on to download the model

learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir='/output/model/')
#Lets find the correct range for learning rate

learn.lr_find()

learn.recorder.plot()
#Lets train the model for 8 cycles

learn.fit_one_cycle(8)
#saving the model 

learn.save('stage-1-50')
learn.unfreeze()
#Lets see if we can fine tune it a bit

learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-3))
learn.save('stage-2-50')