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
from fastai.text import *
from pathlib import Path
folder_path = Path("/kaggle/input/fake-and-real-news-dataset")

os.listdir(folder_path)
folder_path.ls()
true_data = pd.read_csv(folder_path/'True.csv')

fake_data = pd.read_csv(folder_path/'Fake.csv')
true_data.head()
fake_data.head()
print("Shape of the true data df is: ", true_data.shape)

print("Shape of the fake data df is: ", fake_data.shape)
true_data = true_data.assign(is_fake=0);

fake_data = fake_data.assign(is_fake=1);
true_data.head()
fake_data.head()
full_data = true_data.append(fake_data)
full_data.head()
full_data.shape
data = (TextList.from_df(df=full_data, path=folder_path, cols=1)

       .split_by_rand_pct(0.2)

       .label_from_df(cols=4)

       .databunch())
data.show_batch()
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1)
data_title = (TextList.from_df(df=full_data, path=folder_path, cols=0)

       .split_by_rand_pct(0.2)

       .label_from_df(cols=4)

       .databunch())
learn = text_classifier_learner(data_title, AWD_LSTM, drop_mult=0.5,

                                model_dir='/tmp/models')
learn.fit_one_cycle(1)