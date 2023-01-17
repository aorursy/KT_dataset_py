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
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head(50
       )
df.info()
import matplotlib.pyplot as plt

gta5 = df[df.Name == 'Grand Theft Auto V']
gta_sa = df[df.Name == 'Grand Theft Auto: San Andreas']

plt.plot(gta5.Year, gta5.EU_Sales, color = "red", label = "GTA5" )

plt.plot(gta_sa.Year, gta_sa.EU_Sales, color = "green", label = "GTA SA" )
plt.xlabel("Year")
plt.ylabel("EU Sales")
plt.legend()
plt.show()

nin = df[df.Publisher == 'Nintendo']
act = df[df.Publisher =='Activision']
plt.scatter(nin.EU_Sales, nin.JP_Sales, color ='red', label='Nintendo')
plt.scatter(act.EU_Sales, act.JP_Sales, color ='blue', label='Activision')
plt.xlabel("EU_Sales")
plt.ylabel("JP_Sales")
plt.legend()
plt.show()

plt.hist(nin.Year, bins=30)
plt.xlabel("Year")
plt.ylabel("Nintendo-Play")
plt.title("Nin-Hist")
plt.legend()
plt.show()


import seaborn as sns
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)
plt.show()
df.EU_Sales.plot(kind = 'line', color = 'g', label= 'EU', linewidth=1, alpha=1, grid=True, linestyle ='-')
df.JP_Sales.plot(color ='r', label= 'JP', linewidth=1, alpha=0.8, grid=True, linestyle ="-.")
df.NA_Sales.plot(color='blue', label ='NA', linewidth=1, alpha=0.8, grid=True, linestyle =":")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()