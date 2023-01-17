# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i=0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if i%100==0:

            print(os.path.join(dirname, filename))

        i=i+1    



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.vision.gan import *

ganlist=GANItemList.from_folder(path="../input/images/Images/")

databun=ganlist.split_none().label_from_func(noop).transform(tfms=[[crop_pad(size=64, row_pct=(0,1), col_pct=(0,1))], []], size=64, tfm_y=True).databunch()

%matplotlib inline

databun.show_batch()
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=2)

critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=2)

learn = GANLearner.wgan(databun, generator, critic, switch_eval=False,opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

learn.fit(115)
learn.gan_trainer.switch(gen_mode=True)

learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8))
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(16,16))