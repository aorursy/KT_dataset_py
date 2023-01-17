# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe= pd.read_csv('../input/crime.csv')
dataframe.head()
dataframe.describe()
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
dataframe = np.random.rand(10,15)
sns.heatmap(dataframe)
sns.heatmap(dataframe,annot=True)
sns.heatmap(dataframe, vmin=0,vmax=5)
sns.heatmap(dataframe,center=0)
sns.heatmap(dataframe, annot=True)
sns.heatmap(dataframe,vmin=0,vmax=2)
