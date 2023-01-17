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
import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
df.head(10)
df.columns
df.info()
df.corr()
df.plot(kind = "scatter",x="Other_Sales",y = "EU_Sales")
plt.show()
threshold = sum(df.Global_Sales)/len(df.Global_Sales)
print("threshold:",threshold)
df["Global_Sales_level"] = ["successful" if i > threshold else "unsuccessful" for i in df.Global_Sales]
df.loc[:20,["Global_Sales_level","Global_Sales"]]
x=df["Genre"]=="Sports"
df[x]
print(df['Publisher'].value_counts(dropna =False))
df.describe()
import warnings
warnings.filterwarnings("ignore")

df2 = df.head()
date_list = ["1999-01-08","1999-02-08","1999-03-08","1999-03-08","1999-03-16"]
datetime_object = pd.to_datetime(date_list)
df2["date"] = datetime_object
# lets make date as index
df2= df2.set_index("date")
df2 
print(df2.loc["1999-03-16"])
df2.resample("M").mean().interpolate("linear")