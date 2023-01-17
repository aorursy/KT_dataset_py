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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import pandas as pd

data= pd.read_csv("../input/titanic/train_and_test2.csv")
data.head()
data.isnull().sum()
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Fare'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Fare')

ax[0].set_ylabel('')

sns.countplot('Fare',data=data,ax=ax[1])

ax[1].set_title('Fare')

plt.show()

data.groupby(['Sex','Fare'])['Fare'].count()

data.groupby(['Sex','sibsp'])['sibsp'].count()