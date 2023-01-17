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
import pandas as pd

df = pd.read_csv("../input/gold-interest-rate-reg/regression test(1).csv")
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from scipy import stats
df.describe()
sns.boxplot(x = 'interest_rate_pec_change', y = 'gold_returns', data = df)
x = df['interest_rate_pec_change']

y = df['gold_returns']

plt.scatter(x, y)



plt.title('interest_rate_pec_change vs gold_returns')

plt.xlabel('interest_rate_pec_change')

plt.ylabel('gold_returns')

plt.xlim(-0.02,0.02)

plt.ylim(-0.005,0.005)
df.corr()
LM = LinearRegression()

x = df[['interest_rate_pec_change']]

y = df[['gold_returns']]
LM.fit(x,y)
LM.intercept_
LM.coef_
LM.score(x,y)
sns.regplot(x = 'interest_rate_pec_change', y = 'gold_returns', data = df)

plt.ylim(0,)



plt.xlim(-0.02,0.02)

plt.ylim(-0.005,0.005)

plt.xlabel('interest_rate_pec_change')

plt.ylabel('gold_returns')
pearson_coef, p_value = stats.pearsonr(df['interest_rate_pec_change'], df['gold_returns'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  