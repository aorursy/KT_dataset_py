# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/fifa19/data.csv")

df.head()
df.dtypes
import missingno

missingno.matrix(df)
import matplotlib.pyplot as plt

plt.plot(df["Age"],df["Overall"])

plt.show()
import seaborn as sns

sns.regplot(df["Age"],df["Overall"])
sns.regplot(df["BallControl"],df["Overall"])
df.corr()["Overall"]