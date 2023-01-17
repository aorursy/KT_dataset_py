import warnings

import zipfile

from fastai.vision import *

from fastai.metrics import error_rate

from fastai.widgets import *

import pandas as pd

import base64

from IPython.display import HTML

import re



warnings.filterwarnings('ignore') # Suppress warning messages.



%matplotlib inline
os.chdir('/kaggle/input/dogs-vs-cats-redux-kernels-edition')



os.listdir()
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:

    zip_ref.extractall('/kaggle/working/')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:

    zip_ref.extractall('/kaggle/working/')
os.chdir('/kaggle/working/')



os.listdir()
train_fnames = get_image_files('/kaggle/working/train')



len(train_fnames)
train_fnames[:5]
labels = [('cat' if 'cat.' in str(fname) else 'dog') for fname in train_fnames]



labels[:5]
np.random.seed(123) # Ensure reproducibility.

data = ImageDataBunch.from_lists(

    path='/kaggle/working/train', 

    fnames=train_fnames, 

    labels=labels, 

    valid_pct=0.2, # Put 20% of the images in the validation set.

    ds_tfms=get_transforms(flip_vert=False), # Perform data augmentation.

    size=224, # Resize all images to the same size (224px by 224px).

    bs=32 # Set the batch size for training.

).normalize(imagenet_stats) # Normalize all images with ImageNet statistics.
len(data.train_ds), len(data.valid_ds)
data.classes
data.show_batch(rows=3, figsize=(12, 12))
torch.cuda.is_available()
torch.backends.cudnn.enabled
# Make sure Internet is on.

learner = cnn_learner(data, models.resnet50, metrics=error_rate).to_fp16()
learner.lr_find(start_lr=1e-6)
learner.recorder.plot()
max_lr_choice = 5e-4
learner.fit_one_cycle(4, max_lr=max_lr_choice)
learner.recorder.plot_losses()
learner.save('imgsize224-stage1')
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()
accuracy = (interp.confusion_matrix()[0, 0] + interp.confusion_matrix()[1, 1]) / len(data.valid_ds)



accuracy
interp.plot_top_losses(20, figsize=(16, 16))
data_no_split = ImageDataBunch.from_lists(

    path='/kaggle/working/train', 

    fnames=train_fnames, 

    labels=labels, 

    valid_pct=0, # Don't put any images in the validation set.

    ds_tfms=get_transforms(flip_vert=False),

    size=224,

    bs=32

).normalize(imagenet_stats)
%%capture

learner_no_split = cnn_learner(data_no_split, models.resnet50).to_fp16()

learner_no_split.load('imgsize224-stage1')
dataset, file_indices = DatasetFormatter.from_toplosses(learner_no_split)
# Use in interactive mode only. When committing notebook, cleaning isn't possible.

ImageCleaner(dataset, file_indices, Path('/kaggle/working/train'))
# Use in interactive mode only.

cleaned = pd.read_csv('/kaggle/working/train/cleaned.csv')



cleaned.head()
# Source: https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

def create_download_link(df, title="Download CSV file", filename="dogs_vs_cats_cleaned.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload, title=title, filename=filename)

    return HTML(html)
# Use in interactive mode only.

create_download_link(cleaned)



# See 'Download CSV file' link below.
# Make sure Internet is on.

cleaned = pd.read_csv('https://storage.googleapis.com/cleaned-data/dogs_vs_cats_cleaned.csv')



cleaned.head()
np.random.seed(123)

data_cleaned = ImageDataBunch.from_df(

    path='/kaggle/working/train', 

    df=cleaned, 

    valid_pct=0.2, # Put 20% of the images in the validation set.

    ds_tfms=get_transforms(flip_vert=False),

    size=224,

    bs=32

).normalize(imagenet_stats)
learner = cnn_learner(data_cleaned, models.resnet50, metrics=error_rate).to_fp16()
learner.lr_find(start_lr=1e-6)
learner.recorder.plot()
max_lr_choice = 5e-4
learner.fit_one_cycle(4, max_lr=max_lr_choice)
learner.save('imgsize224-stage2')
np.random.seed(123)

data = ImageDataBunch.from_df(

    path='/kaggle/working/train', 

    df=cleaned, 

    valid_pct=0.2,

    ds_tfms=get_transforms(flip_vert=False),

    size=300, # Re-size all images to 300px by 300px.

    bs=32

).normalize(imagenet_stats)
%%capture

learner = cnn_learner(data, models.resnet50, metrics=error_rate).to_fp16()

learner.load('imgsize224-stage2')
learner.lr_find(start_lr=1e-6, end_lr=1e-2, stop_div=False)
learner.recorder.plot()
max_lr_choice = 5e-5
learner.fit_one_cycle(4, max_lr=max_lr_choice)
learner.save('imgsize300-stage1')
learner.unfreeze()

learner.lr_find(start_lr=1e-6, end_lr=1e-2, stop_div=False)
learner.recorder.plot()
learner.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))
learner.save('imgsize300-stage2')
learner = learner.to_fp32() # Convert back to default precision for safe export.

learner.export()
test_fnames = get_image_files('/kaggle/working/test')



len(test_fnames)
test_fnames[:5]
ids = [int(re.findall(r'\d+', str(fname))[0]) for fname in test_fnames]



ids[:5]
learner.path # Location of pickle.
learner = load_learner(path=learner.path, test=test_fnames)
preds, labels = learner.get_preds(ds_type=DatasetType.Test)



preds[:5]
d = {'id': ids, 'label': preds[:, 1]}

submission = pd.DataFrame(data=d)

submission = submission.sort_values(by='id')



submission.head()
submission.to_csv('submission.csv', index=False)