%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate



# Mount to google drive

from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)

root_dir = "/content/gdrive/My Drive/"

base_dir = root_dir + 'fastai-v3/covid19/data'



folder = 'ards'

path = Path(base_dir)

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
bs = 16

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path.ls()
classes = ['pneumocystis','streptococcus','nofinding','covid19','ards','sars']
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c

data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(6)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

# If the plot is not showing try to give a start and end learning rate

# learn.lr_find(start_lr=1e-5, end_lr=1e-1)

learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-04,1e-03))
learn.save('stage-2')
learn.show_results()
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
losses,idxs = interp.top_losses()

interp.plot_top_losses(15, figsize=(15,11))

# doc(interp.plot_top_losses)

# Show images in top_losses along with their prediction, actual, loss, and probability of actual class.

## interp.most_confused(min_val=2)
learn.export()
learn = load_learner(path)
img = open_image(path/'covid19'/'21DDEBFD-7F16-4E3E-8F90-CB1B8EE82828.jpeg')

img
pred_class,pred_idx,outputs = learn.predict(img)

pred_class