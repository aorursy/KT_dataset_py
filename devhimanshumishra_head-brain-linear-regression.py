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
df = pd.read_csv("/kaggle/input/headbrain/headbrain.csv")
#print(df.keys())
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.strip("(cm^3)").str.strip("(grams)")
print(df.keys())
head_size_series = df.head_size
brain_weight_series = df.brain_weight
head_size_series = df.iloc[:, 2:3].values
brain_weight_series = df.iloc[:,3].values
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(head_size_series, brain_weight_series)
m = model.coef_
c = model.intercept_
#print("m = ",m, "c = ", c)
brain_weight_series_predict = model.predict(head_size_series)
#print(brain_weight_series_predict)
import matplotlib.pyplot as plt

plt.scatter(head_size_series, brain_weight_series)
plt.plot(head_size_series, brain_weight_series_predict, c="orange")
plt.show()