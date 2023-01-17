from fastai import *

from fastai.vision import *

from torchvision.models import * 



import os

import matplotlib.pyplot as plt
path = Path("../input/stanford-dogs-dataset/")

path
path.ls()
# path_anno = path/'annotations/Annotations'

path_img = path/'images/Images/'



# path_anno

path_img
# , classes=data.classes[:20]
path_img.ls()
tfms = get_transforms()

# data = ImageDataBunch.from_folder(path_img ,train='.', valid_pct = 0.2,ds_tfms = tfms , size = 227)





np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=227, num_workers=0).normalize(imagenet_stats)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=227, num_workers=0, classes=data.classes[:20]).normalize(imagenet_stats)
# data.normalize(imagenet_stats)
data.show_batch(rows = 3 ,figsize = (7,6))
print(data.classes)

len(data.classes), data.c # data.c = for classification problems its number of classes
learn = create_cnn(data , models.resnet34, metrics = error_rate) 
learn.fit_one_cycle(4)
learn.model_dir = "/kaggle/working"

learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))
learn.model_dir = "/kaggle/working"

learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9 , figsize = (15,11))
interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)
interp.most_confused(min_val = 2) # useful tool
data.classes
img = open_image('../input/test-cc/japanese_spaniel.jpg')

img
classes = data.classes

data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data2 , models.resnet34)

learn.load('/kaggle/working/stage-2')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
prediction = str(pred_class)

prediction[10:]

print("The predicted breed is " + prediction[10:] + '.')