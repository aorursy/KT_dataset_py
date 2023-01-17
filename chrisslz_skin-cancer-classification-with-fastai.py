#the data is deployed twice...

!ls /kaggle/input/skin-cancer-malignant-vs-benign/
from fastai.vision import *

import torchvision
path = Path('/kaggle/input/skin-cancer-malignant-vs-benign/data')

classes = ['malignant','benign']

ImageList.from_folder(path)
tfms = get_transforms(do_flip=True,flip_vert=True)

data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders

        .split_by_folder('train','test')              #How to split in train/valid? -> use the folders

        .label_from_folder()            #How to label? -> depending on the folder of the filenames

        .transform(tfms, size=224)       #Data augmentation? -> use tfms with a size of 64

        .databunch(bs=32))
data
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
# model loading implemented thanks to https://www.kaggle.com/faizu07/kannada-mnist-with-fastai

!mkdir -p /tmp/.cache/torch/checkpoints

!cp /kaggle/input/fastai-pretrained-models/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth

learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."),pretrained=True)
learn.summary() 
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.load('stage-1')

learn.unfreeze()
learn.lr_find(start_lr=1e-9, end_lr=1e-1)
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6, max_lr=slice(1e-04,1e-05)) 
learn.save('stage-2')

#learn.export('skin_classifier.pkl')
learn.load('stage-2');interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(12)