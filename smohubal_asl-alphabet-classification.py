import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = Path("../input/asl_alphabet_train/asl_alphabet_train")

categories = os.listdir(path)



paths =[]

for cat in categories:

    paths.append(Path(f'{path}/{cat}'))

    

def sample_plots(paths : list, categories : list):

    _,axs = plt.subplots(6,5,figsize=(12,12))

    

    n = 0

    for p in paths: 

        for i in [os.listdir(p)[0]]:

            img = open_image(f'{p}/{i}')

            img.show(axs[n%6][n//6], title=f'{categories[n]}')

            n+=1



    plt.tight_layout()

sample_plots(paths, categories)



def class_balance(path, categories):

    t = 0

    class_n = dict()

    for cat in categories:

        class_n[cat] = len(os.listdir(path/cat))

        t += class_n[cat]

    pd.DataFrame(dict(class_n=class_n)).plot.bar()

    return t



t = class_balance(path, categories)



from fastai.torch_core import *

default_device = torch.device('cuda')



id_list = []

for cat in categories:

    id_list += os.listdir(path/cat)



id_list = np.array(id_list)

np.random.shuffle(id_list)

id_list = list(id_list)



tfms = get_transforms(max_rotate=25, do_flip = True, flip_vert = False)

val_idx = id_list[:int(len(id_list) * .22)]



bs = int(t *.005)

data = (ImageList.from_folder(path)

        .split_by_files(valid_names=val_idx)

        .label_from_folder()

        .transform(tfms,size=200)

        .databunch(bs=bs, num_workers=5)

        .normalize(imagenet_stats))
len(categories), data.c
data.device = torch.device('cuda:0')
learn = cnn_learner(data, models.resnet18, wd=.01, model_dir="/tmp/model/", metrics=accuracy)

learn.lr_find()
learn.recorder.plot()
fit_one_cycle(learn, 2, slice(1e-4, 1e-1), moms=(.9, .5))
sample_plots(paths, categories)
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_top_losses(9, figsize=(12,12), heatmap=False)
learn.save('state-01')
learn = learn.load('state-01')