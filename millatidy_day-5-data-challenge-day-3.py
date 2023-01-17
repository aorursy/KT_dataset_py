import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
data = pd.read_csv('../input/cereal.csv')
data.describe()
data.head()
sugars = data['sugars']

sodium = data['sodium']
np.std(sugars)
np.std(sodium)
ttest_ind(sugars, sodium, equal_var = False)
plt.figure()

plt.subplot(2,2,1)

sns.distplot(sugars, kde=False).set_title('Sugar Freq')



plt.figure()

plt.subplot(2,2,2)

sns.distplot(sodium, kde=False).set_title('Sodium Freq')