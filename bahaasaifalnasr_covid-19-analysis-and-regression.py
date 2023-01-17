# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics as st #to get some statics for data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
data
Confirmed=data.Confirmed
Confirmed
Deaths=data.Deaths
Deaths
from matplotlib import pyplot as plt
plt.plot(Confirmed,Deaths,marker='o')

plt.xlabel('Confirmed')

plt.ylabel('Deaths')