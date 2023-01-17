import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
train_data.head().T
train_data.day.value_counts()
train_data.month.value_counts()
import seaborn as sns

sns.set()
p = sns.countplot(train_data.day)
p = sns.lineplot(x = list(range(0, train_data.shape[0])), y = sorted(train_data.day))
p = sns.countplot(train_data.month)
p = sns.lineplot(x = list(range(0, train_data.shape[0])), y = sorted(train_data.month))
import numpy as np



train_data['day_sin'] = np.sin((train_data.day-1)*(2.*np.pi/7))

train_data['day_cos'] = np.cos((train_data.day-1)*(2.*np.pi/7))

train_data['month_sin'] = np.sin((train_data.month-1)*(2.*np.pi/12))

train_data['month_cos'] = np.cos((train_data.month-1)*(2.*np.pi/12))
p = sns.lineplot(x = list(range(0,30)), y = train_data.day_cos[:30])
p = sns.lineplot(x = list(range(0,30)), y = train_data.day_sin[:30])
sample = train_data[:5000] # roughly the first week of the data

ax = sample.plot.scatter('month_sin', 'month_cos').set_aspect('equal')
ax = sample.plot.scatter('day_sin', 'day_cos').set_aspect('equal')