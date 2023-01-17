%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

from pathlib import Path

import os

os.listdir('/kaggle/input/spot-the-mask/')
def random_seed(seed_value):

    import random 

    random.seed(seed_value) # Python

    import numpy as np

    np.random.seed(seed_value) # cpu vars

    import torch

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
train_df = pd.read_csv('/kaggle/input/spot-the-mask/train_labels.csv')

test_df = pd.read_csv('/kaggle/input/spot-the-mask/sample_sub.csv')
train_df.head(3)
test_df['+ACI-image+ACI-'] = test_df['+ACI-image+ACI-'].str.replace(r'\+ACI\-','',regex=True)

test_df.head(2)
data_folder = Path('/kaggle/input/spot-the-mask/')
random_seed(42)



test_img = ImageList.from_df(test_df, path=data_folder, folder='images')



trfm = get_transforms()



train_img = (ImageList.from_df(train_df, path=data_folder, folder='images')

        .split_by_rand_pct(0.1)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=224)

        .databunch(path='.', bs=32, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
train_img.show_batch(rows=3, figsize=(7,6))
print(train_img.classes)

len(train_img.classes), train_img.c
random_seed(42)

learn = cnn_learner(train_img, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
random_seed(42)

learn.fit_one_cycle(10)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(train_img.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
random_seed(42)

learn.freeze_to(-3)

learn.fit_one_cycle(5, wd=0.5)

learn.save('stage-2');
random_seed(42)

learn.freeze_to(-2)

learn.fit_one_cycle(5, wd=0.5)

learn.save('stage-3');
random_seed(42)

learn.freeze_to(-1)

learn.fit_one_cycle(5, wd=0.5)

learn.save('stage-4');
import numpy as np

import pandas as pd
learn.load('stage-4')

log_preds, test_labels = learn.get_preds(ds_type=DatasetType.Test)
log_preds
np.array(log_preds).shape
pred = np.array(log_preds)[:,1]
test_df['+ACI-target+ACI-'] = pred

test_df.head(3)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(test_df)