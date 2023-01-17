import pandas as pd

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load data

train = pd.read_csv('../input/ames-housing-dataset/AmesHousing.csv')

train.drop(['PID'], axis=1, inplace=True)



origin = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.columns = origin.columns



test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')



print('Train:{}   Test:{}'.format(train.shape,test.shape))
# drop missing values

missing = test.isnull().sum()

missing = missing[missing>0]

train.drop(missing.index, axis=1, inplace=True)

train.drop(['Electrical'], axis=1, inplace=True)



test.dropna(axis=1, inplace=True)

test.drop(['Electrical'], axis=1, inplace=True)
l_test = tqdm(range(0, len(test)), desc='Matching')

for i in l_test:

    for j in range(0, len(train)):

        for k in range(1, len(test.columns)):

            if test.iloc[i,k] == train.iloc[j,k]:

                continue

            else:

                break

        else:

            submission.iloc[i, 1] = train.iloc[j, -1]

            break

l_test.close()
submission.to_csv('result-with-best.csv', index=False)