# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.vision.all import *
path = Path('../input/fruits/fruits-360/')
Path.BASE_PATH = path
path.ls()
fruits = get_image_files(path)



im = Image.open(fruits[0])



im.to_thumb(224)
len(fruits)
train = get_image_files(path/'Training')

test = get_image_files(path/'Test')



print(f"Training Images: {len(train)}, Test Images: {len(test)}")
(path/'Training').ls()
train[6000].parent
re.findall(r'([A-Za-z\s]+)(?:\s\d)?$', str(Path(train[6000]).parent))[0]
def my_labeler(o):

    "Extracting Name from Parent Folder without number"

    o = str(Path(o).parent)

    label = re.findall(r'([A-Za-z\s]+)(?:\s\d)?$', o)[0]

    return str(label)
dblock = DataBlock(

                blocks=(ImageBlock, CategoryBlock),

                get_items=get_image_files,

                get_y=my_labeler,

                splitter=RandomSplitter(seed=42, valid_pct=0.2),

                item_tfms=Resize(100),

                batch_tfms=aug_transforms()

)
dls = dblock.dataloaders(path, bs=64)
dls.show_batch()
dls.show_batch(unique=True)
learn = cnn_learner(dls, resnet18, metrics=[error_rate, accuracy])
learn.loss_func
learn.opt_func
learn.lr_find()
learn.fine_tune(epochs=4, freeze_epochs=1, base_lr=1e-2)
learn.show_results()
test[500]
learn.predict(test[500])[0]
test[8452]
learn.predict(test[8452])[0]
test[7500]
learn.predict(test[7500])[0]