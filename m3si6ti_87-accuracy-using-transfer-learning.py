%reload_ext autoreload

%autoreload 2

%matplotlib inline
# start working with fast.ai

from fastai.vision import *

from fastai.metrics import accuracy, error_rate

from fastai import *
# load dataset's location

data_dir='../input/stanford-car-dataset-by-classes-folder/car_data/car_data'

path = Path(data_dir)

# load databunch object

data = ImageDataBunch.from_folder(path, valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False),

                                  size=224,bs=32, num_workers=0).normalize(imagenet_stats)
# visualize data

data.show_batch(3, figsize=(8,8))
# create a conv neural network

learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate], model_dir="/tmp/model/")
# find the best learning rate for this model against provided dataset

learn.lr_find()

learn.recorder.plot()
# select learning rate

lr = 1e-3
# start training using pre-trained model

learn.fit_one_cycle(25, max_lr=lr)
# plot losses

learn.recorder.plot_losses()
# save model

learn.save('model_epoch25_acc87')



# export your model in pickle format

#learn.export('car_model.pkl', )
# interpret and check losses or confusion

interp = learn.interpret()

interp.plot_top_losses(2, figsize=(15,5))