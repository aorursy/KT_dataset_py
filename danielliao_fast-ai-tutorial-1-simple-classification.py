# # not really working

# !pip install --upgrade pip



# !pip install fastai==1.0.46
# load libraries

import fastai.vision as fv
# get dataset

mnist = fv.untar_data(fv.URLs.MNIST_TINY)
# get image transformer

tfms = fv.get_transforms(do_flip=False)
# get data ready for training

data = (fv.ImageItemList.from_folder(mnist)

        .split_by_folder()          

        .label_from_folder()

        .transform(tfms, size=32)

        .databunch()

        .normalize(fv.imagenet_stats))
# check the prepared data

data.show_batch(1)
# see more data in one go

data.show_batch(rows=2, figsize=(4,4))
# create a cnn model with pretrained weights

learn = fv.create_cnn(data, fv.models.resnet18, metrics=fv.accuracy)
# train one epoch

learn.fit_one_cycle(1,1e-2)
# save trained weights

learn.save('mini_train')
# check model performance in examples

learn.show_results(rows=2) 
# see model performance in examples

learn.show_results(ds_type=fv.DatasetType.Train, rows=2, figsize=(8,10))
# import fastai.vision as fv



# # check attached methods and classes

# fv.



# check hidden methods: dunder

fv.__version__



# # check basics info

# fv?



# # check source

# fv??
mnist = fv.untar_data(fv.URLs.MNIST_TINY)



# # check all URLs

# fv.URLs.



# # print the url

# fv.URLs.MNIST_TINY



# # check args

# fv.untar_data?



# # always default?

# # Kaggle kernels

# # Colab notebooks

# # your local



# # check source

# fv.untar_data??



# only time I found to change default

# a specific location to store dataset folder

# !pwd

# dest = fv.Path("/Users/Natsume/Documents/fastai/docs_src/my_tutorials") # or just string 

# mnist = fv.untar_data(fv.URLs.MNIST_TINY, dest=dest)

# mnist.ls()



# (mnist/"valid").ls()
tfms = fv.get_transforms(do_flip=False); tfms



# # check args

# fv.get_transforms?



# # check source

# fv.get_transforms??
data = (fv.ImageItemList.from_folder(mnist) # path list => ImageList

        .split_by_folder()   # ImageList => ItemLists has two ImageLists (train, valid)

        .label_from_folder() # ItemLists => 2 (train and valid) LabelLists, 

                             # train LL has a ImageList x, a CategoryList y; same to valid LL

        .transform(tfms, size=32) # apply transformation to images, size from 28 => 32 too

        .databunch() # LabelList => ImageDataBunch has 2 LLs (train, valid)

                     # Train LL has x as ImageList, y as CategoryList; same to valid LL

        .normalize(fv.imagenet_stats)) # normalize all images by mean and sd of ImageNet



# # class info

# fv.ImageList?



# # class source, check methods

# fv.ImageList??



# # what does `from_folder` do?

# # path list => ImageList

# mnist.ls()

# data = fv.ImageList.from_folder(mnist); data



# # check args 

# fv.ImageList.from_folder?
# # what does `split_by_folder` do?

# # ImageList => ItemLists has two ImageLists (train, valid)

# data = fv.ImageList.from_folder(mnist).split_by_folder(); data



# # check args

# fv.ImageList.split_by_folder?



# # what does `label_from_folder` do?

# # ItemLists => 2 (train and valid) LabelLists, 

# # train LL has a ImageList x, a CategoryList y; same to valid LL

# data = fv.ImageList.from_folder(mnist).split_by_folder().label_from_folder(); data
# # apply transformation to images, size from 28 => 32 too

# data = (fv.ImageList.from_folder(mnist).split_by_folder()

#                 .label_from_folder().transform(tfms, size=32)); data



# # LabelList => ImageDataBunch has 2 LLs (train, valid)

# # Train LL has x as ImageList, y as CategoryList; same to valid LL

# data = (fv.ImageList.from_folder(mnist).split_by_folder()

#                 .label_from_folder().transform(tfms, size=32)

#                 .databunch()); data
# # normalize all images by mean and sd of ImageNet

# data = (fv.ImageList.from_folder(mnist).split_by_folder()

#                 .label_from_folder().transform(tfms, size=32)

#                 .databunch()

#                 .normalize(fv.imagenet_stats)); data
data.show_batch(1)

# data.show_batch?

# data.show_batch(rows=3, ds_type=fv.DatasetType.Train)

# fv.DatasetType.Valid

# fv.DatasetType.Train

# fv.DatasetType.Test

# data.show_batch(rows=3, ds_type=fv.DatasetType.Train, figsize=(5,5)) # (1,5), (5,1)
learn = fv.create_cnn(data, fv.models.resnet18, metrics=fv.accuracy)



# fv.create_cnn?

# fv.create_cnn??

# learn.
learn.fit_one_cycle(1,1e-2)



# # check all args 

# # default to the best practice

# learn.fit_one_cycle?
learn.save('mini_train')



# # check args

# learn.save?



# return_path = True

# learn.save('mini_train', return_path=True)



# # what is model_dir

# learn.model_dir



# # working directory

# !pwd

# learn.model_dir = "/Users/Natsume/Documents/fastai/docs_src/my_tutorials"
learn.show_results(rows=2) 



# # check args

# learn.show_results?



# # try ds_type

# learn.show_results(ds_type = 1) # no effect, not working

# learn.show_results(ds_type = fv.DatasetType.Valid) # right way



# # try rows and figsize

# learn.show_results(ds_type=fv.DatasetType.Train, rows=2, figsize=(8,10))