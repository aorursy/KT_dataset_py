from fastai.vision import *

from fastai.metrics import error_rate,accuracy

import os
bs = 64

Path
path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

path.ls()
img = open_image(path/'val'/'NORMAL'/'NORMAL2-IM-1440-0001.jpeg')

print(img.data.shape)

img.show()
tfms = get_transforms()
np.random.seed(7)

data = ImageDataBunch.from_folder(path,valid = 'val',valid_pct=0.2,size=256,bs =bs,ds_tfms=tfms).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=[error_rate,accuracy], model_dir="../input")
path.ls()
learn.fit_one_cycle(4)
learn.model_dir = "/kaggle/working"
learn.save('stage-1')
learn.unfreeze()
learn.lr_find(stop_div=False, num_it=200)
learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(10,min_grad_lr)
learn.save('stage-2')
learn.validate(learn.data.test_dl)
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
