import pandas as pd

import numpy as np

from scipy.stats import ttest_ind
cereals = pd.read_csv('../input/cereal.csv')
cereals.describe()
cereals.head()
cereals['type'].unique()
cold = cereals.loc[cereals['type'] == 'C']['sugars']

hot = cereals.loc[cereals['type'] == 'H']['sugars']

np.std(cold)
np.std(hot)
ttest_ind(cold, hot, equal_var=False)
import seaborn as sns



sns.distplot(cold).set_title('Cold cereals sugars')
sns.distplot(hot).set_title('Hot cereals sugars')