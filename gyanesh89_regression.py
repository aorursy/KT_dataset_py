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
df=pd.read_csv("/kaggle/input/mtcars/mtcars.csv")
df.describe()
import numpy as np

import statsmodels

import seaborn as sns

from matplotlib import pyplot as plt
import statsmodels.formula.api as sm

reg = sm.ols('mpg ~ hp+qsec+disp', data=df).fit()

reg.summary()
plt.plot(reg.resid)
reg.fittedvalues
pred_val = reg.fittedvalues.copy()

true_val = df['mpg'].values.copy()

residual = true_val - pred_val
residual
fig, ax = plt.subplots(figsize=(6,2.5))

_ = ax.scatter(residual, pred_val)
import scipy as sp

fig, ax = plt.subplots(figsize=(6,2.5))

_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)

r**2