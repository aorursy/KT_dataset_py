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

df = pd.read_csv('/kaggle/input/headbrain/headbrain.csv')
df
df.info()
df.describe()
df.isnull().sum()
import matplotlib.pyplot as plt

df.hist(bins=30)
plt.tight_layout()
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.stats import norm
plt.hist(df['Brain Weight(grams)'], bins=30)
plt.show()
n, bins, patches = plt.hist(df['Brain Weight(grams)'], bins=30, density=1,)

(mu, sigma) = norm.fit(df['Brain Weight(grams)'])

y = norm.pdf(bins,mu,sigma)
plt.plot(bins, y)
(mu, sigma)
(-1.96 * sigma) + mu
z = np.arange(bins[0],bins[-1],0.001)
plt.plot(z, norm.pdf(z,mu,sigma))
cond = (z<(0.675 * sigma) + mu) & (z>(-0.675 * sigma) + mu)
plt.fill_between(z[cond], 0, norm.pdf(z[cond], mu, sigma))
df.columns
import seaborn as sns

sns.jointplot('Head Size(cm^3)', 'Brain Weight(grams)', df, kind='hex')
df.Gender.value_counts()
df['_Gender'] = ['M' if one == 1 else 'F' for one in df.Gender]
df.head(3)
plt.figure(figsize=(8,8))

sns.scatterplot(data=df, x="Head Size(cm^3)", y="Brain Weight(grams)", hue="_Gender")
f, axes = plt.subplots(1, 2)

sns.boxplot(x="_Gender", y="Brain Weight(grams)", data=df, ax=axes[0])
sns.boxplot(x="_Gender", y="Head Size(cm^3)", data=df, ax=axes[1])
plt.tight_layout()
