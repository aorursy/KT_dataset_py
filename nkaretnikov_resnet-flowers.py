%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
LABEL_MAP = {

    0: 'phlox',

    1: 'rose',

    2: 'calendula',

    3: 'iris',

    4: 'leucanthemum maximum',

    5: 'bellflower',

    6: 'viola',

    7: 'rudbeckia laciniata',

    8: 'peony',

    9: 'aquilegia' 

}



# Need to set this to a writable directory.

MODEL_DIR = '/kaggle/working'

DATA_DIR = '../input/flower_images/flower_images'

LABELS = 'flower_labels.csv'

IMAGE_SIZE = 204  # decreasing this negatively affects the results



df = pd.read_csv(f'{DATA_DIR}/{LABELS}')

df.replace(to_replace=LABEL_MAP, inplace=True)

df.head()
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_df(DATA_DIR, df, ds_tfms=tfms, size=IMAGE_SIZE)

data.classes
data.show_batch(rows=5, figsize=(12, 12))
len(data.classes), data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
learn.model_dir = MODEL_DIR

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)
interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=1)
learn.unfreeze()

learn.fit_one_cycle(1)
learn.load('stage-1');  # ; omits the output
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-2))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
data = ImageDataBunch.from_df(DATA_DIR, df, ds_tfms=tfms, size=IMAGE_SIZE, bs=64).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)

learn.model_dir = MODEL_DIR

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=1)
interp.plot_top_losses(4, figsize=(10, 10))
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-3))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=1)
learn.path = Path(MODEL_DIR)

PKL_FILE = Path('resnet-flowers.pkl')

learn.export(file=PKL_FILE)

!ls $MODEL_DIR/$PKL_FILE