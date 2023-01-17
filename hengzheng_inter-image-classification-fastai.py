%reload_ext autoreload

%autoreload 2

%matplotlib inline
!mkdir ./images/

!cp -a ../input/seg_train/seg_train ./images/train

!cp -a ../input/seg_test/seg_test ./images/valid

!cp -a ../input/seg_pred/seg_pred ./images/test
from fastai import *

from fastai.vision import *



import matplotlib.pyplot as plt
data = ImageDataBunch.from_folder(path="./images/", train="train", valid="valid", test="test", ds_tfms=get_transforms(), size=224, bs=64)

data.normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=[accuracy])
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10, 1e-02)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
predictions, targets = learn.get_preds(ds_type=DatasetType.Test)
classes = predictions.argmax(1)

class_dict = dict(enumerate(learn.data.classes))

labels = [class_dict[i] for i in list(classes[:9].tolist())]

test_images = [i.name for i in learn.data.test_ds.items][:9]
plt.figure(figsize=(10,8))



for i, fn in enumerate(test_images):

    img = plt.imread("./images/test/" + fn, 0)

    plt.subplot(3, 3, i+1)

    plt.imshow(img)

    plt.title(labels[i])

    plt.axis("off")
!rm -rf ./images/