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
data = pd.read_csv("/kaggle/input/linearregressiondataset3/linear-regression-dataset.csv")
data.head()
data.isnull().sum()
x=data[['deneyim']]

y=data['maas']
import matplotlib .pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,5))

sns.regplot(x =x,y =y,color='red')
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x,y)

ypred = model.predict(x)
from sklearn.metrics import r2_score

print(r2_score(ypred,y))