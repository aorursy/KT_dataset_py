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
mushrooms_data=pd.read_csv("../input/mushrooms.csv")

mushrooms_data.head()
mushrooms_data['class'].nunique()
mushrooms_data.shape
mushrooms_data.count()
mushrooms_data['class'].value_counts()
mushrooms_data.info()
mushrooms_data.describe()
mushrooms_data['habitat'].value_counts().plot.bar()
pd.plotting.parallel_coordinates(mushrooms_data,"population")