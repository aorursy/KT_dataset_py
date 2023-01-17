# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
data.head(10)
data.info()
data.columns

f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,linecolor="pink",vmin=-1.0,vmax=1.0,linewidths=.5,ax=ax,cmap="coolwarm",fmt=".1f")
plt.scatter(data.Year,data.NA_Sales,color = "red",alpha = 0.3,label = "nasales")
plt.legend()
plt.xlabel("year")
plt.ylabel("sales")
plt.title("SALES IN NA") 
plt.show()
plt.scatter(data.Year,data.EU_Sales,color = "green",alpha = 0.2,label = "eusales")
plt.legend()
plt.xlabel("year")
plt.ylabel("sales")
plt.title("SALES IN EU") 
plt.show()
plt.scatter(data.Year,data.JP_Sales,color = "blue",alpha = 0.1,label = "jpsales")
plt.legend()
plt.xlabel("year")
plt.ylabel("sales")
plt.title("SALES IN JP") 
plt.show()
plt.scatter(data.Year,data.Global_Sales,color = "yellow",alpha = 0.1,label = "GlobalSales")
plt.legend()
plt.xlabel("year")
plt.ylabel("sales")
plt.title("SALES IN GLOBAL") 
plt.show()

data[data["Global_Sales"]>30]


data.Year = data.Year
data.head()
a = data.groupby("Platform")
a["Global_Sales"].sum()