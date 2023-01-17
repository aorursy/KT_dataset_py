%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import*

from fastai.metrics import error_rate
import os
testpath = "../input/test1/test1/"

trainpath = "../input/train/train/"
testpath
trainpath
torch.cuda.is_available()
torch.backends.cudnn.enabled
#fnames = (os.listdir(f'{trainpath}'))
fnames = get_image_files(trainpath)
fnames[:5]
labels = [('cat' if 'cat' in str(x) else 'dog') for x in fnames]
labels[:5]
data = ImageDataBunch.from_lists(trainpath, fnames, ds_tfms=get_transforms(), size=224, bs=64, labels = labels)

data.classes
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/")
learn.fit_one_cycle(4)
learn.lr_find()

learn.recorder.plot()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.save('model-1')
#learn.unfreeze()

#learn.fit_one_cycle(2, max_lr=slice(1e-3,1e-2))
#learn.load('model-1')
#learn.fit_one_cycle(2, max_lr=slice(1e-3,1e-2))
#learn.lr_find()

#learn.recorder.plot()
#learn.load('model-1')
#learn.fit_one_cycle(2)
#learn.lr_find()

#learn.recorder.plot()
#learn.fit_one_cycle(2, max_lr=8e-07)
#learn.lr_find()

#learn.recorder.plot()
#learn.save('model-2')
#learn.load('model-1')
import numpy as np

import pandas as pd
test_images=get_image_files(testpath)

test_images[:5]
fnames[:5]
submission = pd.DataFrame(os.listdir("../input/test1/test1/"),columns = ['id'])
submission['label']=0
submission
count = 0
count
for imgpath in test_images:

    img = open_image(imgpath)

    pred = learn.predict(img)

    if str(pred[0]) != 'dog':

        submission['label'][count]=0

        count = count+1

    else:

        submission['label'][count]=1

        count = count+1
img = open_image('../input/test1/test1/1112.jpg')

pred = learn.predict(img)
pred
pred[1]
submission[['id','label']].to_csv('sampleSubmission.csv',index=False)