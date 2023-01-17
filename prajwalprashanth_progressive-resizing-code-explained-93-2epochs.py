from fastai import *

from fastai.vision import *
!pwd #current directory
!ls ../
!ls ../input
!ls ../input/train
!ls ../input/train/train
!ls ../input/train/train | wc -l #number of classes in train set
!ls ../input/test
!ls ../input/test/test | wc -l #number of images in dataset
path = Path('../input') #parent path



#low res data

data_64 = (ImageList.from_folder(path/'train') #have specified the train directory as it has a child dir named train which contains all the classes in folders

                .split_by_rand_pct(0.1, seed=33) #since there is no validation set, we are taking 10% of the train set as validation set

                .label_from_folder()#to label the images based on thier folder name/class

                .add_test_folder('..'/path/'test')#came out of the current directory and specified where test set is at, as it doesnt follow the imagenet style of file structure

                .transform(get_transforms(), size=64)#using the default transforms and initial size of 64x64

                .databunch(bs=256)#batch size of 256, be cautious of OOM error when you increase the size of the image decrease the batchsize to be able to fit in the memory

                .normalize(imagenet_stats))#normalizing to the imagenet stats



#high res data

data_256 = (ImageList.from_folder(path/'train')

                .split_by_rand_pct(0.1, seed=33)

                .label_from_folder()

                .add_test_folder('..'/path/'test')

                .transform(get_transforms(), size=256)

                .databunch(bs=64)

                .normalize(imagenet_stats))
data_64 #verifying the no. of images, split of different sets and you can observe test has no labels
data_64.c #verifying the no. of classes
learn = cnn_learner(data_64, #training on low res first 

                    models.resnet18, #loading the resenet18 arch with pretrained weights

                    metrics=accuracy, 

                    model_dir='/tmp/model/') #specifying a different directory as /input is a read-only directory and will throw an error while using lr_find()
learn.lr_find() #finds the change in loss with respect to the learning rate

learn.recorder.plot()#plots that change
learn.fit_one_cycle(1, 1e-2)
learn.data = data_256 #loading the high res images

learn.unfreeze() #unfreezing the inital layers
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-4,1e-3))