# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MeanShift



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
X = pd.DataFrame(df.iloc[:, 4:13])
normalised = StandardScaler()

X = normalised.fit_transform(X)
ms = MeanShift(0.5)

ms_result=ms.fit_predict(X)
X = pd.DataFrame(X)

plt.scatter(X.iloc[:,7], X.iloc[:,5],  c=ms_result)

plt.show() 