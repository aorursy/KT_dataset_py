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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import warnings
import pandas as pd

df = pd.read_csv("../input/residue_buildig_1_sub_meter_2.csv")

df=df.rename(columns={"-11.09121856947948": "ei"})

df.head()
df.columns




df['ei_square'] = np.square(df['ei'])

sum_of_squared_residuals = df.sum()["ei_square"]

df['ei_minus_1'] = df['ei'].shift()

df.dropna(inplace=True)

df['ei_sub_ei_minus_1'] = df['ei'] - df['ei_minus_1']

df['square_of_ei_sub_ei_minus_1'] = np.square(df['ei_sub_ei_minus_1'])

sum_of_squared_of_difference_residuals = df.sum()["square_of_ei_sub_ei_minus_1"]

dw = sum_of_squared_of_difference_residuals/sum_of_squared_residuals

print('DW_Stats = '+str(dw))