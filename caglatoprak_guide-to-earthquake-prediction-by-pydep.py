# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pd.options.display.precision=15

train=pd.read_csv("../input/train.csv",nrows=10000000,dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train.head()
train.rename({"acoustic_data":"sinyal","time_to_failure":"time"},axis="columns",inplace=True)

train.head()
tests=os.listdir("../input/test")

print(tests[0:3])

len(tests)
sample_submission=pd.read_csv("../input/sample_submission.csv")

sample_submission.head(2)
print(len(sample_submission))
train.describe().T
fig,ax=plt.subplots(1,2,figsize=(20,5))

ax[0].set_title("Signal distribution")

ax[1].set_title("Signal distribution without peaks");



sns.kdeplot(train.sinyal,ax=ax[0],shade=True);

low = train.sinyal.mean() - 3 * train.sinyal.std()

high = train.sinyal.mean() + 3 * train.sinyal.std() 

sns.distplot(train.loc[(train.sinyal >= low) & (train.sinyal <= high), "sinyal"].values,ax=ax[1]);
sns.boxplot(x="sinyal",data=train)