# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/credit-card-fraud-detection/creditcard.csv')
df.head()
classification=df.Class.value_counts(sort= True)

classification
LABELS=('Normal','Fraud')

plt.figure(figsize=(10,5))

classification.plot(kind = 'bar',rot=0)

plt.xlabel('Class')

plt.ylabel('Number of observations')

plt.xticks(range(2), LABELS)

plt.title('Transaction class Distribution')

plt.show

y=df['Class']

X=df.drop('Class',axis=1)
y.value_counts()
from imblearn.combine import SMOTETomek

smk = SMOTETomek()
X_res,y_res = smk.fit_sample(X,y)
oversampling=y_res.value_counts()

oversampling
LABELS=('Normal','Fraud')

plt.figure(figsize=(10,5))

oversampling.plot(kind = 'bar',rot=0)

plt.xlabel('Class')

plt.ylabel('Number of observations')

plt.xticks(range(2), LABELS)

plt.title('Oversampled Transaction class Distribution')

plt.show
from imblearn.under_sampling import NearMiss

nm = NearMiss()
X_ndr,y_ndr = nm.fit_sample(X,y)
undersampling=y_ndr.value_counts()

undersampling
LABELS=('Normal','Fraud')

plt.figure(figsize=(10,5))

undersampling.plot(kind = 'bar',rot=0)

plt.xlabel('Class')

plt.ylabel('Number of observations')

plt.xticks(range(2), LABELS)

plt.title('Undersampled Transaction class Distribution')

plt.show