# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statsmodels.stats.weightstats import _tconfint_generic as stm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/water.txt", sep="\t")

data
data.groupby("location").describe()
mor = data.groupby("location")["mortality"]

mean = mor.mean()

#mor.size()

pd.DataFrame(stm(mean, mor.std() / np.sqrt(mor.size()), mor.size() - 1, 0.05, "two-sided")).T
har = data.groupby("location")["hardness"]

mean = har.mean()

#mor.size()

pd.DataFrame(stm(mean, har.std() / np.sqrt(har.size()), har.size() - 1, 0.05, "two-sided")).T