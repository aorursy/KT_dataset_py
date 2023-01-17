from fastai.vision import *

import os



path = Path(r'../input/samples/')

print(os.listdir(path/'samples')[:10])
from IPython.display import Image

Image(filename='../input/samples/samples/bny23.png')
def label_from_filename(path):

    label = [char for char in path.name[:-4]]

    return label
data = (ImageList.from_folder(path)

        .split_by_rand_pct(0.2)

        .label_from_func(label_from_filename)

        .transform(get_transforms(do_flip=False))

        .databunch()

        .normalize()

       )

data.show_batch(3)
acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = learn = cnn_learner(data, models.resnet18, model_dir='/tmp', metrics=acc_02)

lr_find(learn)

learn.recorder.plot()
lr = 5e-2

learn.fit_one_cycle(5, lr)
import copy

losses = copy.deepcopy(learn.recorder.losses)
learn.unfreeze()

lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(15, slice(1e-3, lr/5))
losses += learn.recorder.losses



fig, ax = plt.subplots(figsize=(14,7))

ax.plot(losses, linewidth=2)

ax.set_ylabel('loss', fontsize=16)

ax.set_xlabel('iteration', fontsize=16)

plt.show()