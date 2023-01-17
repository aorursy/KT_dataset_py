# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df1 = df.drop(["Rank"],axis=1)

df.head(100)
df.info()
df.corr()
df.columns
df.plot(kind = 'scatter', x='Year', y='Platform', alpha = 0.5, color = 'red', linewidth = 10000 )

plt.title("Pl")

plt.legend()

plt.show()
Number = df[df.Platform == "2600"]

SNES = df[df.Platform == "SNES"]

GC = df[df.Platform == "GC"]

PSV = df[df.Platform == "PSV"]



plt.scatter(Number.Year, Number.EU_Sales, color = "purple", label = "2600")

plt.scatter(SNES.Year, SNES.EU_Sales, color = "red", label = "SNES")

plt.scatter(GC.Year, GC.EU_Sales,  color = "blue", label = "GC")

plt.scatter(PSV.Year, PSV.EU_Sales, color = "green", label = "PSV")

plt.title("Platform EU sales")

plt.legend()

plt.show()
plt.scatter(Number.Year, Number.JP_Sales, color = "purple", label = "2600")

plt.scatter(SNES.Year, SNES.JP_Sales, color = "red", label = "SNES")

plt.scatter(GC.Year, GC.JP_Sales,  color = "blue", label = "GC")

plt.scatter(PSV.Year, PSV.JP_Sales, color = "green", label = "PSV")

plt.title("Platform JP sales")

plt.legend()

plt.show()
plt.scatter(Number.Year, Number.NA_Sales, color = "purple", label = "2600")

plt.scatter(SNES.Year, SNES.NA_Sales, color = "red", label = "SNES")

plt.scatter(GC.Year, GC.NA_Sales,  color = "blue", label = "GC")

plt.scatter(PSV.Year, PSV.NA_Sales, color = "green", label = "PSV")

plt.title("Platform NA sales")

plt.legend()

plt.show()
plt.scatter(Number.Year, Number.Other_Sales, color = "purple", label = "2600")

plt.scatter(SNES.Year, SNES.Other_Sales, color = "red", label = "SNES")

plt.scatter(GC.Year, GC.Other_Sales,  color = "blue", label = "GC")

plt.scatter(PSV.Year, PSV.Other_Sales, color = "green", label = "PSV")

plt.title("Platform Other sales")

plt.legend()

plt.show()
plt.scatter(Number.Year, Number.Global_Sales, color = "purple", label = "2600")

plt.scatter(SNES.Year, SNES.Global_Sales, color = "red", label = "SNES")

plt.scatter(GC.Year, GC.Global_Sales,  color = "blue", label = "GC")

plt.scatter(PSV.Year, PSV.Global_Sales, color = "green", label = "PSV")

plt.title("Platform Global sales")

plt.legend()

plt.show()
df.columns
Gb_sales_threshold = sum(df.Global_Sales[:10])/len(df.Global_Sales[:10])

EU_sales_threshold = sum(df.EU_Sales[:10])/len(df.EU_Sales[:10])

NA_sales_threshold = sum(df.NA_Sales[:10])/len(df.NA_Sales[:10])

JP_sales_threshold = sum(df.JP_Sales[:10])/len(df.JP_Sales[:10])

Other_sales_threshold = sum(df.Other_Sales[0:10])/len(df.Other_Sales[:10])

print("Gb_sales_threshold",Gb_sales_threshold)

print("EU_sales_threshold",EU_sales_threshold)

print("NA_sales_threshold",NA_sales_threshold)

print("JP_sales_threshold",JP_sales_threshold)

print("Other_sales_threshold",Other_sales_threshold)

df["GS_Ratio"] = [ "High" if i > Gb_sales_threshold else "Low" for i in df.Global_Sales]

df["EU_Ratio"] = [ "High" if i > EU_sales_threshold else "Low" for i in df.EU_Sales]

df["NA_Ratio"] = [ "High" if i > NA_sales_threshold else "Low" for i in df.NA_Sales]

df["JP_Ratio"] = [ "High" if i > JP_sales_threshold else "Low" for i in df.JP_Sales]

df["Other_Ratio"] = [ "High" if i > Other_sales_threshold else "Low" for i in df.Other_Sales]

df.loc[:10,["GS_Ratio","Global_Sales","EU_Ratio","EU_Sales","NA_Ratio","NA_Sales","JP_Ratio","JP_Sales","Other_Ratio","Other_Sales","Name"]]