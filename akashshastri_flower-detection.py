# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/train.csv')
df.head()

from fastai.vision import *

from fastai import *
from pathlib import Path

path = Path('/kaggle/input/flower-recognition-he/he_challenge_data/data/')

path
sz = 128

bs = 32

tfms = get_transforms(do_flip=True,

                      max_rotate=15,

                      max_warp=0.,

                      max_lighting=0.1,

                      p_lighting=0.3

                     )

src = (ImageList.from_df(df=df

                         ,path=path/'train'

                         ,cols='image_id'

                         , suffix = '.jpg'

                         #,convert_mode='L'

                        ) 

        .split_by_rand_pct(0) 

        .label_from_df(cols='category') 

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )
from fastai.callbacks import *
learn = cnn_learner(data, base_arch=models.resnet101, metrics = [accuracy],

                    callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=2)],path = '/kaggle/working/', model_dir = '/kaggle/working/'

                    ).mixup()
learn.fit_one_cycle(4)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(20, max_lr=slice(1e-5,1e-4))
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.save('stg-1')
bs = 32 

sz=256

tfms = get_transforms(do_flip=True,

                      max_rotate=15,

                      max_warp=0.,

                      max_lighting=0.1,

                      p_lighting=0.3

                     )

src = (ImageList.from_df(df=df

                         ,path=path/'train'

                         ,cols='image_id'

                         , suffix = '.jpg'

                         #,convert_mode='L'

                        ) 

        .split_by_rand_pct(0.15) 

        .label_from_df(cols='category') 

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )
learn.load('stg-1')
learn.data = data 
learn.freeze()

learn.fit_one_cycle(4)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(20, max_lr=slice(7e-6, 1e-5), wd = 1e-1)

learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.save('stg-2')
bs = 32 

sz=320

tfms = get_transforms(do_flip=True,

                      max_rotate=15,

                      max_warp=0.,

                      max_lighting=0.1,

                      p_lighting=0.3

                     )

src = (ImageList.from_df(df=df

                         ,path=path/'train'

                         ,cols='image_id'

                         , suffix = '.jpg'

                         #,convert_mode='L'

                        ) 

        .split_by_rand_pct(0.15) 

        .label_from_df(cols='category') 

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='reflection') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )
learn.load('stg-2')
learn.data = data 
learn.freeze()

learn.fit_one_cycle(4)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(20, max_lr=slice(5e-5, 1e-5), wd = 1e-1)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.save('stg-3')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
sample_df = pd.read_csv('/kaggle/input/flower-recognition-he/he_challenge_data/data/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,'/kaggle/input/flower-recognition-he/he_challenge_data/data/',folder='test',suffix='.jpg'))
preds,y = learn.TTA(ds_type=DatasetType.Test)
labelled_preds = []
for pred in preds:

    labelled_preds.append(int(np.argmax(pred))+1)
sample_df.category = labelled_preds

sample_df.groupby('category').count()
sample_df.to_csv('submission.csv',index=False)
sample_df = sample_df.sort_values(by = ['image_id'], ascending = [True])
learn.export('dense161.pkl')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sample_df)


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


interp.most_confused(min_val=2)