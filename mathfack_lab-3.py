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
data = pd.read_csv("/kaggle/input/water.txt", sep="\t")

data.head()
data.loc[data.location == "South"].describe()
data.loc[data.location == "North"].describe()
from statsmodels.stats.weightstats import _tconfint_generic

import numpy as np



mort_mean_north = data.loc[data.location == "North"].mortality.mean()

mort_mean_std_north = data.loc[data.location == "North"].mortality.std() / np.sqrt(data.loc[data.location == "North"].mortality.shape[0])

print("95% interval north:", _tconfint_generic(mort_mean, mort_mean_std, data.loc[data.location == "North"].mortality.shape[0]-1, 0.05, "two-sided"))
mort_mean = data.loc[data.location == "South"].mortality.mean()

mort_mean_std = data.loc[data.location == "South"].mortality.std() / np.sqrt(data.loc[data.location == "South"].mortality.shape[0])

print("95% interval south:", _tconfint_generic(mort_mean, mort_mean_std, data.loc[data.location == "South"].mortality.shape[0]-1, 0.05, "two-sided"))
hard_mean = data.loc[data.location == "North"].hardness.mean()

hard_mean_std = data.loc[data.location == "North"].hardness.std() / np.sqrt(data.loc[data.location == "North"].hardness.shape[0])

print("95% interval north:", _tconfint_generic(hard_mean, hard_mean_std, data.loc[data.location == "North"].hardness.shape[0]-1, 0.05, "two-sided"))
hard_mean = data.loc[data.location == "South"].hardness.mean()

hard_mean_std = data.loc[data.location == "South"].hardness.std() / np.sqrt(data.loc[data.location == "South"].hardness.shape[0])

print("95% interval south:", _tconfint_generic(hard_mean, hard_mean_std, data.loc[data.location == "South"].hardness.shape[0]-1, 0.05, "two-sided"))