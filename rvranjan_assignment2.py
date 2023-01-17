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
import pandas as pd

auto_mpg = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
auto_mpg
import seaborn as sns
auto_mpg.columns
sns.boxplot(auto_mpg["weight"])

sns.boxplot(auto_mpg["mpg"])
sns.boxplot(auto_mpg["displacement"])
auto_mpg.dtypes
auto_mpg.horsepower.unique()
data=auto_mpg.horsepower.replace('?',np.NaN)
data.unique()