!pip install -q efficientnet_pytorch
import numpy as np

import pandas as pd
from fastai import *

from fastai.utils import *

from fastai.vision import *

from fastai.callbacks import *

from pathlib import Path

import seaborn as sns

import matplotlib.pyplot as plt

import PIL

from torch.utils import model_zoo



%matplotlib inline
from efficientnet_pytorch import EfficientNet
import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir('.'))
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
print('Make sure cuda is installed:', torch.cuda.is_available())

print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
print(os.listdir('../input'))
hack_path = Path('../input')
# Load train dataframe

train_df = pd.read_csv(hack_path/'train/train.csv')

test_df = pd.read_csv(hack_path/'test_ApKoW4T.csv')

sample = pd.read_csv(hack_path/'sample_submission_ns2btKE.csv')
def get_data(bs, size):

    data = ImageDataBunch.from_df(df=train_df, path=hack_path/'train', folder='images',

                                  bs=bs, size=size, valid_pct=0.1, 

                                  resize_method=ResizeMethod.SQUISH, 

                                  ds_tfms=get_transforms(max_lighting=0.4, max_zoom=1.2, 

                                                         max_warp=0.2, max_rotate=20, 

                                                         xtra_tfms=[flip_lr()]))

    test_data = ImageList.from_df(test_df, path=hack_path/'train', folder='images')

    data.add_test(test_data)

    data.normalize(imagenet_stats)

    return data
data = get_data(bs=48, size=224)
data.show_batch(rows=3, figsize=(10,8))
model_name = 'efficientnet-b3'
def get_model(pretrained=True, **kwargs):

    model = EfficientNet.from_pretrained(model_name)

    model._fc = nn.Linear(model._fc.in_features, data.c)

    return model
learn = Learner(data, get_model(), 

                metrics=[FBeta(), accuracy],

                callback_fns=[partial(SaveModelCallback)],

                wd=0.1,

                path = '.').mixup()
learn.lr_find()

learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(20, min_grad_lr)
learn.recorder.plot_losses()
learn.recorder.plot_lr(show_moms=True)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(20, slice(min_grad_lr))
unfrozen_validation = learn.validate()

print("Final model validation loss: {0}".format(unfrozen_validation[0]))
learn.save('efficientnet-unfrozen', return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(15, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp.most_confused(min_val=2)
probability, classification = learn.TTA(ds_type=DatasetType.Test)
probability.argmax(dim=1)[:10]
(probability.argmax(dim=1) + 1).unique()
sample.category = probability.argmax(dim=1) + 1
sample.category.value_counts()
sample.head()
sample.to_csv('submission_efficientnetb3_kaggle.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(sample)