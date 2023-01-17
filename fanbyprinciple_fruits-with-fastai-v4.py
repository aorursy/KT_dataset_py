# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install fastbook

from fastai import *
from fastai.vision.all import *
root_dir = '../input/fruits/fruits-360'
path = Path(root_dir + '/Test')
classes = []
for dirname,_,_ in os.walk(path):
    name = dirname.split('/')[-1].strip() 
    if (name != 'Test'):
        classes.append(name)

print(classes[:5])
print("Number of classes : ", len(classes))
fns =  get_image_files(path)
fns[:5]
#failed = verify_images(fns)
#failed.map(Path.unlink)
fruits = DataBlock (
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct=0.2, seed=42),
    get_y = parent_label,
    item_tfms = Resize(128)
)
dls = fruits.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
fruits = fruits.new(item_tfms=RandomResizedCrop(224, min_scale=0.5),
                   batch_tfms=aug_transforms())

dls = fruits.dataloaders(path)
dls.show_batch(max_n=16, nrows=4, unique=True)
learn = cnn_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(4)
interp.vocab
# checking on a random fruit form test
learn.predict('../input/fruits/fruits-360/Test/Apple Granny Smith/321_100.jpg')
learn.export()