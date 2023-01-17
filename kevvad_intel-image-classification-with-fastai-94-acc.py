%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy
!mkdir data
!mkdir data/model_data

# create symbolic link to intel data
!cp -rs /kaggle/input/intel-image-classification/ data

!ls data
data_path = Path("data/intel-image-classification/")
data_path.ls()
# adjust according to your GPU memory
batch_size = 128
# load data
# we start with a very low resolution to train faster initially
# later we use progressive resizing to train more accurately

data = (ImageDataBunch
        .from_folder(data_path, 
                     train='seg_train', valid='seg_test', test='seg_pred', 
                     ds_tfms = get_transforms(),
                     seed = 7,
                     num_workers=6,
                     size=64,
                     bs=batch_size)
       )
data
data.classes
data.show_batch(5, figsize=(15, 10))
# using mixed precision often helps in training faster and also somtimes leads to a better generalization
learner = cnn_learner(data, models.resnet50, metrics=accuracy) #.to_fp16()
learner.lr_find()
learner.recorder.plot(suggestion=True)
lr = 3e-3 # fastai uses this as default anyways
learner.fit_one_cycle(10, slice(lr))
learner.recorder.plot_losses()
learner.recorder.plot_lr()
learner.recorder.plot_metrics()
learner.model_dir = Path("data/model_data/")
learner.save('0_resnet50_bs128_64x64_freeze');
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)
lr
learner.fit_one_cycle(5, max_lr=slice(1e-6, lr/10))
learner.save('1_resnet50_bs128_64x64_unfreeze');
data2 = (ImageDataBunch
        .from_folder(data_path, 
                     train='seg_train', valid='seg_test', test='seg_pred', 
                     ds_tfms = get_transforms(),
                     seed = 7,
                     num_workers=6,
                     size=150,  # train with 128/150
                     bs=batch_size)
       )
# use full size data now
learner.data = data2
learner.freeze()
learner.lr_find()
learner.recorder.plot()
lr = 1e-4
learner.fit_one_cycle(5, slice(lr))
learner.save('2_resnet50_bs128_150x150_freeze');
learner.recorder.plot_lr()
learner.recorder.plot_losses()
learner.recorder.plot_metrics()
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)
lr
# put some high LR to learn a bit more faster
learner.fit_one_cycle(10, slice(1e-5, lr/2))
learner.save('3_resnet50_bs128_150x150_unfreeze');
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
interp.plot_top_losses(12)
interp.most_confused()
predictions, targets = learner.get_preds(ds_type=DatasetType.Test)
predictions
classes = predictions.argmax(1)
class_dict = dict(enumerate(learner.data.classes))
labels = [class_dict[i] for i in list(classes[:16].tolist())]
test_images = [i.name for i in learner.data.test_ds.items][:16]
plt.figure(figsize=(20,15))

for i, fn in enumerate(test_images):
    img = plt.imread(data_path/'seg_pred'/'seg_pred'/fn, 0)
    plt.subplot(4, 4, i+1)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")