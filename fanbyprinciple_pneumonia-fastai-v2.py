

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



root_dir = "../input/chest-xray-pneumonia/chest_xray"
!dir {root_dir}
!pip install fastbook
from fastbook import *

from fastai.vision.widgets import *
path = Path(root_dir +'/test')
path.ls()
data = DataBlock( blocks=(ImageBlock, CategoryBlock),

                get_items=get_image_files,

                get_y = parent_label,

                item_tfms = Resize(224),

                splitter = RandomSplitter(valid_pct=0.2,seed=42)

                )



dls = data.dataloaders(path)

# we need to think about how to incorporate validation set here
dls.valid.show_batch(max_n=4, nrows=1)



# It is better to spend some more time here maybe working with the data

# and finding out what lighting to use, but I feel xrays look more or 

# less the same.
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(16,nrows=8)
learn.show_results()
preds, _ =  learn.get_preds()

preds.shape
test_items = get_image_files(path/'test')

dl = learn.dls.test_dl(test_items, rm_type_tfms=1, bs=64)
y_pred , _ = learn.get_preds(dl=dl)

thresh = 0.5

yhat_test = [' '.join([learn.dls.vocab[i] for i,p in enumerate(pred) if p > thresh]) 

             for pred in y_pred.numpy()]

y_pred.numpy()
y_test = list(map(lambda x:x.parents[0].name, test_items))
results = pd.DataFrame({'target': y_test, 'pred': yhat_test})
accuracy = results[results.target == results.pred].shape[0]/ results.shape[0]; accuracy