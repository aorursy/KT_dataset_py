import numpy as np

import pandas as pd

import os

from fastai.vision import *
path = Path('../input')
path.ls()
path_img = path/'train'
get_image_files(path_img)[:5]
np.random.seed(42)

size = 224

bs = 64

num_workers = 0  # set this to 0 to prevent kernel from crashing

pat = r'/([^/.]+).\d+.jpg$'
tfms = get_transforms()                              #Do standard data augmentation

data = (ImageItemList.from_folder(path_img)          #Get the training images from the train dir

        .random_split_by_pct()                       #Randomly split off 20% of the images to form validation set

        .label_from_re(pat)                          #Label by applying the regex to the filenames

        .add_test_folder('../test')                  #Add a test set using the test dir

        .transform(tfms, size=size)                  #Pass in data augmentation

        .databunch(bs=bs, num_workers=num_workers)   #Create ImageDataBunch

        .normalize(imagenet_stats))                  #Normalize using imagenet stats
print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet50, metrics=accuracy, model_dir='/tmp/models')
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp.most_confused(min_val=2)
preds, y = learn.get_preds(ds_type=DatasetType.Test)
dog_preds = preds[:,1]
submission = pd.DataFrame({'id':os.listdir('../input/test'), 'label':dog_preds})
submission['id'] = submission['id'].map(lambda x: x.split('.')[0])
submission['id'] = submission['id'].astype(int)
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)