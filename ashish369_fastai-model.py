# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *
!ls ../input/plant-seedlings-classification/train
path = Path('../input/plant-seedlings-classification/train')
path.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".",test='../test', valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3,fig_size=(7,8))
data.classes, data.c , len(data.train_ds) , len(data.valid_ds) , len(data.test_ds)
learn= cnn_learner(data,models.resnet50,metrics=error_rate)
learn.fit_one_cycle(4)
learn.model_dir='/kaggle/working/'
learn.save('stage_1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5,slice(1e-3))
learn.save('stage_2')
learn.load('stage_2');
interp = ClassificationInterpretation.from_learner(learn)


interp.plot_confusion_matrix(figsize=(10,10))
interp.plot_top_losses(4,figsize=(10,10))
preds,y=learn.get_preds(ds_type=DatasetType.Test)
preds = np.argmax(preds, axis = 1)

preds_classes = [data.classes[i] for i in preds]
submission = pd.DataFrame({ 'file': os.listdir('../input/plant-seedlings-classification/test'), 'species': preds_classes })

submission.to_csv('test_classification_results.csv', index=False)
submission