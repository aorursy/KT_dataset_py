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
df = pd.read_csv("../input/winemag-data-130k-v2.csv")
df.info()
df.describe()
series = df["price"]
print(type(series))
dataFrame = df[["price"]]
print(type(dataFrame))
import matplotlib.pyplot as plt
df.plot(kind="line", color="blue", linewidth=1, alpha=0.3, grid=True)
plt.show()
firstCondition = df[np.logical_and(df["points"] > 88, df["points"] < 91)]
print(type(firstCondition))
firstCondition.plot(kind="scatter", x='price', y='points', linewidth=1, alpha=0.5, grid=True, label='Fiyat Performans Grafiği')
plt.legend()
plt.show()
for key, value in firstCondition[0:10].iterrows():
    print(key, " : ", value)
    
averagePrice = df.price.mean()
print("Average Price is ", averagePrice)
df["price_status"] = ["bigger than average" if averagePrice > each else "lower than average" for each in df.price]
df.loc[:20, ["price", "price_status"]]