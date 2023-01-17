import seaborn as sns # visualization library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind # t-test function

data = pd.read_csv('../input/cereal.csv')

data.describe()
data.head()
ttest_ind(data['calories'], data['rating'], equal_var=False)
sns.distplot(data['calories'],kde= False)

sns.distplot(data['rating'],kde= False)