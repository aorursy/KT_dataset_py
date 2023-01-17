# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from fastai.tabular import *

%matplotlib inline

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/hack-ml/Dataset/Train.csv')

test = pd.read_csv('../input/hack-ml/Dataset/Test.csv')
print('Train set shape:', train.shape)

print('Test set shape:', test.shape)

print('NaN in Train:',train.isnull().values.any())

print('NaN in Test:',test.isnull().values.any())

print('Train set overview:')

display(train.head())
f, ax = plt.subplots(figsize=(6, 6))

ax = sns.countplot(x="netgain", data=train, label="Label count")

sns.despine(bottom=True)
del train['id']

del test['id']
dep_var='netgain'

cat = ['realtionship_status', 'industry', 'genre', 'targeted_sex', 'airtime', 'airlocation',

       'expensive', 'money_back_guarantee']

cont = ['average_runtime(minutes_per_week)','ratings']

procs= [Categorify,Normalize]



inception = TabularList.from_df(test,cat_names=cat, cont_names=cont , procs=procs)

data = (TabularList.from_df(train, cat_names=cat, cont_names=cont , procs=procs)

                .split_subsets(train_size=0.8, valid_size=0.2, seed=34)

                .label_from_df(cols=dep_var)

                .add_test(inception)

                .databunch())
learn = tabular_learner(data, layers=[200,100],metrics=accuracy)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit(1,lr=1e-3)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1,max_lr=1e-8)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.predict(inception[0])[0]