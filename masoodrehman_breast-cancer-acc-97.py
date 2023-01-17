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
from fastai.tabular import *

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.columns
df.drop(["Unnamed: 32"],axis=1,inplace=True)
df.head()
cont_names = [ 'radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']
dep_var = "diagnosis"
procs = [Normalize]
df = df.sample(frac=1).reset_index(drop=True)
df.tail()
data = (TabularList.from_df(df,cont_names=cont_names,procs=procs)

       .split_subsets(train_size=0.7,valid_size=0.3,seed=42)

       .label_from_df(cols=dep_var)

       .databunch())
learn = tabular_learner(data,[400,200],metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8,1e-03)
learn.save("stage-1")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,1e-03)
learn.load("stage-1")