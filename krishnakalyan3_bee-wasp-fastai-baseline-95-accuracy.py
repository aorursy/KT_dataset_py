!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

!pip install fastai==2.0.9
import random, os

import numpy as np

import torch

from fastai.vision.all import *
files = glob.glob('../input/bee-vs-wasp/kaggle_bee_vs_wasp/*/*.jpg')

f, plots = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(15, 7),  constrained_layout=True)

im_plot = []



for j in files:

    im = Image.open(np.random.choice(files))

    if im.size == (320, 245):

        im_plot.append(im)

    if len(im_plot)==11:

        break

        

for i in range(10):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(im_plot[i])
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
path = Path('/kaggle/input/bee-vs-wasp/kaggle_bee_vs_wasp'); 
metadata = pd.read_csv(path/'labels.csv', index_col=0); metadata.head()

metadata['path'] = metadata['path'].apply(lambda x:x.replace('\\', '/'))
#metadata['fname'] = metadata['path']

dls = ImageDataLoaders.from_df(metadata, path, item_tfms=Resize(412),seed=0,

                              bs=32, num_workers=4, valid_col='is_validation', label_col="label")
dls.show_batch()
print(dls.vocab); print(dls.c)
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/").to_fp16()
learn.lr_find()
learn.fit_one_cycle(50, lr_max=1e-2, cbs=EarlyStoppingCallback(patience=3))
learn.unfreeze()
learn.fit_one_cycle(50, lr_max=slice(2e-7, 1e-4), cbs=EarlyStoppingCallback(patience=3))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
preds, _ = learn.get_preds(); preds.shape