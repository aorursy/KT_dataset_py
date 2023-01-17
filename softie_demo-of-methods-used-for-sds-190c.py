%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path = untar_data(URLs.PETS); path
path.ls()
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
fnames[:5]
np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')
data = ImageDataBunch.from_name_re(path_img, fnames, pat, valid_pct=0.2, ds_tfms=get_transforms(), size=224, bs=bs,
                                  ).normalize(imagenet_stats)
# ImageDataBunch - class, performs normalization, regularization, and splits data into training and validation
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes)
learn = create_cnn(data, models.resnet18, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, 1e-2) # the first number is the number of epochs, the second number is the learning rate
learn_152 = create_cnn(data, models.resnet152, metrics=error_rate)
learn_152.lr_find()
learn_152.recorder.plot()
learn_152.fit_one_cycle(3, 3e-3)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)