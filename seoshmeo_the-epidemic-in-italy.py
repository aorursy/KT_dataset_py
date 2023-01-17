# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/covid-19-italy-updated-regularly/national_data.csv")

df_provincia = pd.read_csv("/kaggle/input/covid-19-italy-updated-regularly/provincial_data.csv")

df_region = pd.read_csv("/kaggle/input/covid-19-italy-updated-regularly/regional_data.csv")
df
plt.grid()              

plt.plot(df.total_positive_cases)

plt.plot(df.new_currently_positive)
df.total_positive_cases

total_cas = df.total_positive_cases[15]

total_death = df.death[15]

CFR = total_death / total_cas * 100

CFR