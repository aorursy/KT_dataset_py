# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import sklearn as sk

df = pd.read_csv('../input/california-housing-prices/housing.csv')
df


df.refine = df.drop('ocean_proximity',axis = 1)

df.refine = df.refine.dropna(axis = 0)
df.refine
X = df.refine.drop('median_house_value', axis = 1)

Y = df.refine['median_house_value']
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X,Y)
model.predict([[-121.24,39.37,16.0,2785.0,616.0,1387.0,530.0,2.3886]])