# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kabggle/python docker image: https://github.com/kaggle/docker-python

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
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()
data.info()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(data['Class'])

plt.title('Countplot')

plt.show()
data.Class.value_counts()
from imblearn.over_sampling import SMOTE
data.corr()
f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
data.shape
data['Amount'] = StandardScaler(data['Amount'])
data.head()