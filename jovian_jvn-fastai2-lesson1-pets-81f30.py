from fastai2.vision.all import *

import jovian

# from nbdev.showdoc import *



set_seed(2)
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
help(untar_data)
path = untar_data(URLs.PETS); path
Path.BASE_PATH = path # display all paths relative to dataset root

path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames
dls = ImageDataLoaders.from_name_re(

    path, fnames, pat=r'(.+)_\d+.jpg$', item_tfms=Resize(460), bs=bs,

    batch_tfms=[*aug_transforms(size=224, min_scale=0.75), Normalize.from_stats(*imagenet_stats)])
dls.show_batch(max_n=9, figsize=(7,6))
print(dls.vocab)

len(dls.vocab),dls.c
learn = cnn_learner(dls, resnet34, metrics=error_rate).to_fp16()
learn.model
learn.fit_one_cycle(4)
import multiprocessing

multiprocessing.cpu_count()
!nvidia-smi
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(dls.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.unfreeze()

learn.fit_one_cycle(2, lr_max=slice(1e-6,1e-4))
dls = ImageDataLoaders.from_path_re(path_img, fnames, pat=r'(.+)_\d+.jpg$', item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=bs//2,

                                     batch_tfms=[*aug_transforms(size=299, max_warp=0), Normalize.from_stats(*imagenet_stats)])
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(3, lr_max=slice(1e-6,1e-4))
learn.load('stage-1-50');
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
path = untar_data(URLs.MNIST_SAMPLE); path
tfms = aug_transforms(do_flip=False)

data = ImageDataLoaders.from_folder(path, batch_tfms=tfms, size=26, bs=bs)
data.show_batch(max_n=9, figsize=(5,6))
learn = cnn_learner(data, resnet18, metrics=accuracy)

learn.fit(2)
df = pd.read_csv(path/'labels.csv')

df.head()
data = ImageDataLoaders.from_csv(path, batch_tfms=tfms, size=28)
data.show_batch(max_n=9, figsize=(5,6))

data.vocab
data = ImageDataLoaders.from_df(df, path=path, batch_tfms=tfms, size=24)

data.vocab
fn_paths = [path/name for name in df['name']]; fn_paths[:2]
pat = r"/(\d)/\d+\.png$"

data = ImageDataLoaders.from_path_re(path, fn_paths, pat=pat, batch_tfms=tfms, size=24)

data.vocab
data = ImageDataLoaders.from_path_func(path, fn_paths, batch_tfms=tfms, size=24,

        label_func = lambda x: '3' if '/3/' in str(x) else '7')

data.vocab
labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]

labels[:5]
data = ImageDataLoaders.from_lists(path, fn_paths, labels=labels, batch_tfms=tfms, size=24)

data.vocab
jovian.commit(message="intro md", project="fastai2-lesson1-pets")