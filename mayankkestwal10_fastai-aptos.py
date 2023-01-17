%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



def seed_everything(seed=1358):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
PATH = Path('../input/aptos2019-blindness-detection/')

train_images = PATH/'train_images'

test_images = PATH/'test_images'



df_train = pd.read_csv(PATH/'train.csv')

df_test = pd.read_csv(PATH/'test.csv')
df_train.head()
df_train['id_code'] = df_train['id_code'].apply(lambda x : str(x) + '.png')
df_test['id_code'] = df_test['id_code'].apply(lambda x : str(x) + '.png')
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360.0,max_lighting=0.2,max_zoom=1.1)
data = ImageDataBunch.from_df(df=df_train,path=PATH,folder='train_images',ds_tfms=get_transforms(),valid_pct=0.2,seed=42,size=256,num_workers=4).normalize(imagenet_stats)
data
kappa = KappaScore()

kappa.weights = "quadratic"
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
learn = cnn_learner(data,models.resnet50,metrics=[kappa], model_dir=Path("/kaggle/working/"),path=Path("."))
learn.fit_one_cycle(4)
learn.recorder.plot_losses()
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,max_lr=slice(1e-8,1e-6))
learn.save('stage-2')
learn.load('stage-2')