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
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=2.5)

import missingno as msno



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.isna().sum()
train.describe()
train.shape
full_data = [train,test]
msno.matrix(df=train,figsize=(8,8))
msno.bar(df=train, figsize=(8,8))
f,ax = plt.subplots(1,2, figsize=(14,8))

train['Survived'].value_counts().plot.pie(ax=ax[0], autopct='%1.1f%%')

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Count plot - Survived')

ax[1].set_ylabel('')

plt.show()
import pandas_profiling as pdp
pdp.ProfileReport(train)