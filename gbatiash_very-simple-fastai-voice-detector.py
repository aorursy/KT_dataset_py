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


from fastai.tabular import * 

from fastai import *

from pathlib import Path

import pandas as pd

# path = untar_data(URLs.ML_SAMPLE)
path = Path('/kaggle/input/voicegender/')

dataframe = pd.read_csv(path/'voice.csv')

dep_var='label'

cont_names = cont_cat_split(dataframe,max_card=20, dep_var=dep_var)[0]

procs = [Normalize]
data = (TabularList.from_df(dataframe, path=path, cont_names=cont_names, procs=procs)

        .split_by_rand_pct(0.20)

        .label_from_df(cols=dep_var)

        .databunch())

learn = tabular_learner(data, layers=[4000,2000], metrics= accuracy)

learn.model_dir='/kaggle/working/'

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,1.5e-4)