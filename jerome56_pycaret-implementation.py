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
!pip install pycaret
# Loading Data into the Data Frame



df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

df.head()
# Dropping Id and Date column

df = df.drop(['id','date'],axis=1)

df.head()
# Importing module and initializing setup

from pycaret.regression import *

reg1 = setup(data = df, target = 'price',normalize = True)
# comparing all models

compare_models()
# creating catboost model

catboost = create_model('catboost')
# tuning Catboost  model

tuned_catboost = tune_model('catboost', n_iter = 50, optimize = 'mae')