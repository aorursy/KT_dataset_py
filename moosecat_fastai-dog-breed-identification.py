import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



#Updating to follow flow in Course v3

#https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb
%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This file contains all the main external libs we'll use - fastai v1

from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate
path = Path("/kaggle/input/dog-breed-identification/")

# 32 when testing 224 when for real - better way from course is to reduce the batch size to 32

#sz=32 

bs = 16

sz=224

TMP_PATH = "/tmp/tmp"

MODEL_PATH = "/tmp/model/"

arch = 'resnet34'

comp_name = "dog_breed"
path.ls()
label_df = pd.read_csv(path/'labels.csv')
#what does the csv file look like id is the file name (minus .jpg), breed is the classification

label_df.head()
# GPU required

torch.cuda.is_available()
torch.backends.cudnn.enabled
# Fix to enable Resnet to live on Kaggle - creates a writable location for the models

cache_dir = os.path.expanduser(os.path.join('~', '.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

   # print("directory created :" .cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

  #  print("directory created :" . cache_dir)
#copying model to writable location

#cd /kaggle/working

shutil.copy("/kaggle/input/resnet34/resnet34.pth", "/tmp/.torch/models/resnet34-333f7ec4.pth")
#crashes with too many workers (remove numworkers while testing with a small image size ~ 64), keep for the bigger images for accuracy

tfms = get_transforms(do_flip=True, max_zoom=1.1)



data = ImageDataBunch.from_csv(path, ds_tfms=tfms, folder="train", test='test', suffix='.jpg', bs=bs, size=sz, num_workers=0).normalize(imagenet_stats) 

#data = ImageDataBunch.from_csv(PATH, ds_tfms=tfms, folder="train", test='test', suffix='.jpg', size=sz)
data.show_batch(rows=3, figsize=(7,6))
#label_df.pivot_table(index='breed', aggfunc=len).sort_values('id', ascending=False) - show the summary from the table

#To save space:

print(data.classes)

len(data.classes),data.c
#img = plt.imread(f'{PATH}train/{label_df.iloc[0,0]}.jpg')

img = plt.imread(path/f'train/{label_df.iloc[0,0]}.jpg')

plt.imshow(img);
img.size
#Look at the distribution of the size of the images

size_d = {}

for index, row in label_df.iterrows():

    size_d[index] = PIL.Image.open(path/f'train/{row[0]}.jpg').size
row_sz,col_sz = list(zip(*size_d.values()))
row_sz=np.array(row_sz);col_sz=np.array(col_sz)
row_sz[:5]
col_sz[:5]
plt.hist(row_sz[row_sz<1000])
plt.hist(col_sz[col_sz<1000])
len(data.classes), data.classes[:5]
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=MODEL_PATH)
learn.fit_one_cycle(4)
#learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
#learn.load('stage-1');

learn.lr_find()
learn.recorder.plot()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-5))
#learn.fit_one_cycle(4,max_lr=(1e-2, (1e-2)/2, (1e-2)/3))

#learn.fit_one_cycle(1,max_lr=(1e-2, (1e-2)/2, (1e-2)/3)) # for testing

#learn.recorder.plot_losses()
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_top_losses(9, figsize=(14,14))
predictions = learn.get_preds(ds_type=DatasetType.Test)
test_df = pd.read_csv(path/'sample_submission.csv')
submission_columns = data.classes.copy()

submission_columns.insert(0, 'id')

df = pd.DataFrame(columns=submission_columns)
for idx in range(len(predictions[0])):

#for idx in range(3):

    probs = predictions[0][idx].tolist()

    formatted_probs = ["%.17f" % member for member in probs]

    filename = [test_df['id'][idx]]

    df.loc[idx] = filename + formatted_probs
df.head()
# Submission

df.to_csv(f"sub_{comp_name}_{arch}.csv", index=False)