%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
path='../input/aerial-cactus-identification/'

path_img_train = path + 'train/train/'

path_img_test = path + 'test/test/'
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_csv(path_img_train, csv_labels='../../train.csv', ds_tfms=tfms, size=28)
data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
learn.unfreeze()
learn.fit_one_cycle(2)
submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
preds=[]

for file in submission.id:

    img = open_image(path_img_test+file)

    pred_class,_,_=learn.predict(img)

    preds.append(pred_class)
preds
#submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

submission.iloc[:,1] = preds
submission.head()
submission.to_csv("submission.csv", index=False)

print('Save submission', datetime.now(),)