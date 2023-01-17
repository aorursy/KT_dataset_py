%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

from pathlib import Path

from ipywidgets import IntProgress

from IPython.display import display
path = '/kaggle/input/lego-brick-images'

imagePath = path +'/dataset'
#Test if the dataset images are found

fnames = get_image_files(imagePath)

fnames[:4]
# Test the regular expression to filter out the classification name. 

# The space at the end is trimmed during import into ImageDataBunch, so don't care.

import re

re.search(r'([^/]+) ', fnames[0].name)[0] 
data = ImageDataBunch.from_name_re(imagePath, 

                                   fnames, 

                                   r'/([^/]+) ', 

                                   ds_tfms=get_transforms(), 

                                   size=224, 

                                   bs=64

                                  ).normalize(imagenet_stats)
#Check the number of training anf validation items.

#The validation.txt file is not used as input, the ImageDataBunch does this.

len(data.train_ds.x.items), len(data.valid_ds.x.items)
#Check the data classes

print([len(data.classes), data.classes])
#Be sure to enable under Kaggle NoteBook Settings: 'Internet' to On and 'GPU' to On

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/kaggle/output')
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(10, slice(6e-3), pct_start=0.9)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))

learn.freeze()
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
fig, ax = plt.subplots(1,2)

ax[0].imshow(plt.imread(f'{imagePath}/3046 roof corner inside tile 2x2 007R.png'));

ax[1].imshow(plt.imread(f'{imagePath}/3003 brick 2x2 000L.png'));
#Determine the error rate with one camera as verification. 

#This must be equal to the last outcome of training epoch.



prg = IntProgress(min=0, max=len(data.valid_ds.x.items)) # instantiate the progress bar

display(prg) # display the progress bar



err = 0

for f in data.valid_ds.x.items:

    cat = f.name[:-9]

    pred_class,pred_idx,outputs = learn.predict(open_image(f))

    pred_cat = learn.data.classes[pred_class.data.item()]

    if pred_cat != cat:

        err += 1

    prg.value += 1

    

print(f'Error rate with one camera: {err/len(data.valid_ds.x.items)}')
fnamesR = [f for f in data.valid_ds.x.items if f.name[-5:] == 'R.png']

fnamesL = [f for f in data.valid_ds.x.items if f.name[-5:] == 'L.png']

print([len(fnamesR), len(fnamesL)])
if len(fnamesR) < len(fnamesL):

    suffix = 'L'

    fnames2 = fnamesR

else:

    suffix = 'R'

    fnames2 = fnamesL

print([suffix, len(fnames2)])
#Determine the error rate with two cameras



prg = IntProgress(min=0, max=len(fnames2)) # instantiate the progress bar

display(prg) # display the progress bar



err = 0

for fA in fnames2:

    fB = Path(f'{imagePath}/{fA.name[:-5]}{suffix}.png')

    cat = fA.name[:-9]

    pred_classA,pred_idxA,outputsA = learn.predict(open_image(fA))

    pred_catA = learn.data.classes[pred_classA.data.item()]

    pred_classB,pred_idxB,outputsB = learn.predict(open_image(fB))

    pred_catB = learn.data.classes[pred_classB.data.item()]

    outputs = outputsA+outputsB

    arr = outputs.numpy()

    maxval = np.amax(arr)

    maxind = np.where(arr == maxval)[0][0]

    if data.classes[maxind] != cat:

        err += 1

    prg.value += 1

print(f'Error validation set with two cameras: {err/len(fnames2)}')