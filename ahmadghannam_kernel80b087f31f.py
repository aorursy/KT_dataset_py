from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
path = Path('/kaggle/input/waste-classification-data/dataset/DATASET')
path_local = Path('/kaggle/input/wasteclassificationlocal')
data_local.show_batch(rows=10,figsize=(10,10))
data_local = ImageDataBunch.from_folder(
    path_local,
    valid = "TEST",
    ds_tfms=get_transforms(do_flip=False),
    size = 128,
    bs=32,
    valid_pct=0.2,
).normalize(imagenet_stats)
print(f'Classes: \n {data_local.classes}')
data_local.show_batch(rows=10,figsize=(10,10))
data = ImageDataBunch.from_folder(
    path,
    train = "TRAIN",
    valid = "TEST",
    ds_tfms=get_transforms(do_flip=False),
    size = 128,
    bs=32,
    valid_pct=0.2,
    num_workers=0
).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=10,figsize=(10,10))
learn_vgg16 = create_cnn(data, models.vgg16_bn,metrics=accuracy,model_dir='/tmp/model/')
learn_vgg16.lr_find()
learn_vgg16.recorder.plot()
learn_vgg16.fit_one_cycle(5)
learn_resnet50 = create_cnn(data, models.resnet50,metrics=accuracy,model_dir='/tmp/model/')
learn_resnet50.lr_find()
learn_resnet50.recorder.plot()
learn_resnet50.fit_one_cycle(5)
learn_alexnet = create_cnn(data, models.alexnet,metrics=accuracy,model_dir='/tmp/model/')
learn_alexnet.lr_find()
learn_alexnet.recorder.plot()
learn_alexnet.fit_one_cycle(5)
learn_vgg16.recorder.plot_losses()
pd.DataFrame(np.array(learn_vgg16.recorder.losses).tolist()).to_csv("/kaggle/working/learn_vgg16.csv")
learn_resnet50.recorder.plot_losses()
pd.DataFrame(np.array(learn_resnet50.recorder.losses).tolist()).to_csv("/kaggle/working/learn_resnet50.csv")
learn_alexnet.recorder.plot_losses()
pd.DataFrame(np.array(learn_alexnet.recorder.losses).tolist()).to_csv("/kaggle/working/learn_alexnet.csv")
learn_vgg16.save("/kaggle/working/learn_vgg16")
learn_resnet50.save("/kaggle/working/learn_resnet50")
learn_alexnet.save("/kaggle/working/learn_alexnet")
learn_vgg16.data = data_local
learn_resnet50.data = data_local
learn_alexnet.data = data_local
preds_vgg16, y_vgg16 = learn_vgg16.get_preds(ds_type=data_local.valid_ds)
(np.argmax(preds_vgg16, axis=1) == y_vgg16).sum(), y_vgg16.shape[0]
preds_resnet50, y_resnet50 = learn_resnet50.get_preds(ds_type=data_local.valid_ds)
(np.argmax(preds_resnet50, axis=1) == y_resnet50).sum(), y_resnet50.shape[0]
preds_alexnet, y_alexnet = learn_alexnet.get_preds(ds_type=data_local.valid_ds)
(np.argmax(preds_alexnet, axis=1) == y_alexnet).sum(), y_alexnet.shape[0]
inter = ClassificationInterpretation.from_learner(learn)
inter.plot_top_losses(9,figsize=(20,20))