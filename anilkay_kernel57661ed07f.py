# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision.gan import *

from fastai.vision import *

bs=128

size=64

data=(GANItemList.from_folder("/kaggle/input/pokemon-images-dataset/", noise_sz=100)

                    .split_none()

                    .label_from_func(noop)

                    .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)

                    .databunch(bs=bs)

                    .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)

critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)

learn = GANLearner.wgan(data, generator, critic, switch_eval=False,

                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

learn.fit(50)
learn.fit(50)
learn.fit(150)
learn.gan_trainer.switch(gen_mode=True)

learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(12,12))