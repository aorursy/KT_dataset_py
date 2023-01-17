import os

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import math
data = pd.read_csv('../input/loan.csv', low_memory=False)

data.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)



list(data)
plt.hist(data['loan_amnt'])

plt.show()