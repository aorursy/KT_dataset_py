%reload_ext autoreload

%autoreload 2

%matplotlib inline
from pathlib import Path

from fastai.vision import *

from fastai.metrics import error_rate
#path = untar_data('https://www.kaggle.com/techsash/waste-classification-data', <destination>)
path = Path('../input/waste-classification-data/dataset/DATASET'); 

path
pathTest = path/'TEST'

pathTrain = path/'TRAIN'

pathTest.ls()
numberOfFiles = (pathTest/'O').ls()

numberOfFiles[:5]
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), valid_pct=0.2, size=224).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.model_dir = "/kaggle/working"

learn.save('stage-1', return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.path = Path("/kaggle/working")

learn.export()
testImageList = ImageList.from_folder(pathTest/'O')

image = open_image(testImageList.items[0])

image
model = load_learner(learn.path)

className, label, probability = model.predict(image)
className
label
probability