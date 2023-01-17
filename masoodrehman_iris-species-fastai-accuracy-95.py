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
from fastai import *

from fastai.tabular import *

from fastai.metrics import *
path = Path("../working")
df = pd.read_csv("../input/iris/Iris.csv")
df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
df.columns
dep_var = "Species"

cont_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
procs = [Normalize]
data = (TabularList.from_df(df,cont_names=cont_names,procs=procs)

       .split_subsets(train_size=0.6,valid_size=0.4,seed=42)

       .label_from_df(cols=dep_var)

       .databunch())
learn = tabular_learner(data,layers=[200,100],metrics=accuracy)
learn.recorder.plot()
learn.fit_one_cycle(10,1e-02)
learn.unfreeze()
learn.recorder.plot()
learn.fit_one_cycle(7,1e-02)