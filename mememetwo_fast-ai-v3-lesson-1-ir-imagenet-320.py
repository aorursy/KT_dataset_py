%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
help(untar_data)
path = untar_data(URLs.IMAGENETTE_320); path
path.ls()
path_img = path/'train'

print("Here are the labels...  I don't know what n02979186 translates to, but let's find out.")

path_img.ls()

path_img.ls()

import os

os.rename(str(path_img) + '/n01440764', str(path_img) + '/Trench_Fish')

os.rename(str(path_img) + '/n02102040', str(path_img) + '/English_Springer_Dog')

os.rename(str(path_img) + '/n02979186', str(path_img) + '/Cassette_Player')

os.rename(str(path_img) + '/n03000684', str(path_img) + '/Chainsaw')

os.rename(str(path_img) + '/n03028079', str(path_img) + '/Church')

os.rename(str(path_img) + '/n03394916', str(path_img) + '/French_Horn')

os.rename(str(path_img) + '/n03417042', str(path_img) + '/Garbage_Truck')

os.rename(str(path_img) + '/n03425413', str(path_img) + '/Gas_Pump')

os.rename(str(path_img) + '/n03445777', str(path_img) + '/Golf_Ball')

os.rename(str(path_img) + '/n03888257', str(path_img) + '/Parachute')

path_img.ls()
help(get_image_files)

fnames = get_image_files(path_img, True, True)

fnames[:5]
np.random.seed(2)

pat = re.compile(r'/([^/]+)/[^/]+_\d+.JPEG$')
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,10))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(8)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(18,18), heatmap=True)
interp.plot_confusion_matrix(figsize=(18,18), dpi=60)
interp.most_confused(min_val=1)
learn.unfreeze()

learn.fit_one_cycle(2)

learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2)

learn.load('stage-1')
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))

interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(18,18), heatmap=True)
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),

                                   size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4)
learn.save('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(18,18), heatmap=False)

interp.plot_top_losses(9, figsize=(18,18), heatmap=True)