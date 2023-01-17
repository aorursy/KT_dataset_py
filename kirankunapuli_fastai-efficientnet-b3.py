import numpy as np

import pandas as pd



from fastai.utils import *

from fastai.vision import *

from fastai.callbacks import *

from pathlib import Path

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
!pip install -q efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
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
hack_path = Path('../input')
train_df = pd.read_csv(hack_path/'train.csv')

test_df = pd.read_csv(hack_path/'sample_submission.csv')
def get_data(bs, size):

    data = ImageDataBunch.from_df(df=train_df, path=hack_path/'train', folder='train',

                                  bs=bs, size=size, valid_pct=0.1, 

                                  resize_method=ResizeMethod.SQUISH, 

                                  ds_tfms=get_transforms(do_flip=True, flip_vert=True,

                                                         max_lighting=0.2, max_zoom=1.1, 

                                                         max_warp=0.2, max_rotate=10))

    test_data = ImageList.from_df(test_df, path=hack_path/'test', folder='test')

    data.add_test(test_data)

    data.normalize(imagenet_stats)

    return data
data = get_data(bs=64, size=128)
data.show_batch(rows=3, figsize=(5,5))
model_name = 'efficientnet-b3'
def get_model(pretrained=True, **kwargs):

    model = EfficientNet.from_pretrained(model_name)

    model._fc = nn.Linear(model._fc.in_features, data.c)

    return model
learn = Learner(data, get_model(), 

                metrics=[AUROC(), FBeta(), accuracy],

                callback_fns=[partial(SaveModelCallback)],

                wd=0.1,

                path = '.')
learn.lr_find()

learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(10, slice(min_grad_lr))
learn.recorder.plot_losses()
learn.recorder.plot_lr(show_moms=True)
validation = learn.validate()

print("Final model validation loss: {0}".format(validation[0]))
learn.save('efficientnet-cactus', return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(3,3), dpi=80)
interp.most_confused(min_val=2)
probability, _ = learn.TTA(ds_type=DatasetType.Test)
probability.argmin(dim=1)[:10]
probability.numpy()[:, 0]
test_df.has_cactus = probability.numpy()[:, 0]
test_df.head()
test_df.to_csv('submission.csv', index=False)