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
data = pd.read_excel('/kaggle/input/Consumer.xlsx')

print(data.shape)

print()

print(data.head())
data.describe()
import statsmodels

import statsmodels.api as sm

import statsmodels.stats.api as sms



from statsmodels.formula.api import ols
formula = '''Amount_Charged ~ Income

'''

model = ols(formula, data)

results = model.fit()

print(results.summary())
formula = '''Amount_Charged ~ HH_Size

'''

model = ols(formula, data)

results2 = model.fit()

print(results2.summary())
from statsmodels.iolib.summary2 import summary_col

dfoutput = summary_col([results,results2],stars=True)

print(dfoutput)
formula = '''Amount_Charged ~ HH_Size + Income

'''

model = ols(formula, data)

results3 = model.fit()

print(results3.summary())
dfoutput = summary_col([results,results2, results3],stars=True)

print(dfoutput)
estimate = 1304.91 + (356.2959 * 3) + (33.133 * 40)

print('The amount we estimate using a house hold of 3 and income of 40,000 is', round(estimate, 2))