# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/GooglePlayStoreBEFORE.csv')

dataset.head()
dataset.shape



dataset.describe()
dataset.boxplot()
dataset.boxplot(column=['Rating'], return_type='axes');

dataset.hist()
dataset.info(

)
dataset.isnull()
dataset.isnull().sum()
dataset[:[dataset.Reviews>10000]]
threshold = len(dataset)*0.1

dataset.dropna(thresh=threshold,axis=1,inplace=True)

dataset.shape
