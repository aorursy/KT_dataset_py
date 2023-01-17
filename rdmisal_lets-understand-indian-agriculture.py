# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

agri=pd.read_csv('../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv')
agri.head()

# Any results you write to the current directory are saved as output.
x=agri['Years']
y=agri['EMP_AGR']
z=agri['GDP_AGR']
agri.describe().T
import matplotlib.pyplot as plt

from scipy import stats

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
slope
import seaborn as sns
sns.regplot(x,y,label="Agriculture Employment").set_title("Agriculture Employment")

sns.regplot(x,z,label="Agriculture GDP").set_title("Agriculture GDP")
from scipy import stats

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x,z)
slope
data1=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (1).csv')
data1.head()
data1=pd.DataFrame(data=data1)
data1.describe().T
data1.Crop.unique()
Arhar=data1[0:5]
COTTON=data1[6:10]
GRAM=data1[10:15]
GROUNDNUT=data1[15:20]
MAIZE=data1[20:25]
MOONG=data1[25:30]
PADDY=data1[30:36]
RAPESEEDAndMUSTARD=data1[35:40]
SUGARCANE=data1[40:45]
WHEAT=data1[46:49]

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))

x=Arhar['State']
y=Arhar['Cost of Cultivation (`/Hectare) A2+FL']
z=Arhar['Cost of Cultivation (`/Hectare) C2']
a=Arhar['Cost of Production (`/Quintal) C2']
b=Arhar['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("Arhar Production Variation", fontsize=16, fontweight='bold')
plt.show()


plt.figure(figsize=(10,8))
x=COTTON['State']
y=COTTON['Cost of Cultivation (`/Hectare) A2+FL']
z=COTTON['Cost of Cultivation (`/Hectare) C2']
a=COTTON['Cost of Production (`/Quintal) C2']
b=COTTON['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("COTTON Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=GRAM['State']
y=GRAM['Cost of Cultivation (`/Hectare) A2+FL']
z=GRAM['Cost of Cultivation (`/Hectare) C2']
a=GRAM['Cost of Production (`/Quintal) C2']
b=GRAM['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("GRAM Production Variation", fontsize=16, fontweight='bold')
plt.show()




plt.figure(figsize=(10,8))
x=GROUNDNUT['State']
y=GROUNDNUT['Cost of Cultivation (`/Hectare) A2+FL']
z=GROUNDNUT['Cost of Cultivation (`/Hectare) C2']
a=GROUNDNUT['Cost of Production (`/Quintal) C2']
b=GROUNDNUT['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("GROUNDNUT Production Variation", fontsize=16, fontweight='bold')
plt.show()




plt.figure(figsize=(10,8))
x=MAIZE['State']
y=MAIZE['Cost of Cultivation (`/Hectare) A2+FL']
z=MAIZE['Cost of Cultivation (`/Hectare) C2']
a=MAIZE['Cost of Production (`/Quintal) C2']
b=MAIZE['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("MAIZE Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=MOONG['State']
y=MOONG['Cost of Cultivation (`/Hectare) A2+FL']
z=MOONG['Cost of Cultivation (`/Hectare) C2']
a=MOONG['Cost of Production (`/Quintal) C2']
b=MOONG['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("MOONG Production Variation", fontsize=16, fontweight='bold')
plt.show()



plt.figure(figsize=(10,8))
x=PADDY['State']
y=PADDY['Cost of Cultivation (`/Hectare) A2+FL']
z=PADDY['Cost of Cultivation (`/Hectare) C2']
a=PADDY['Cost of Production (`/Quintal) C2']
b=PADDY['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("PADDY Production Variation", fontsize=16, fontweight='bold')
plt.show()


plt.figure(figsize=(10,8))
x=RAPESEEDAndMUSTARD['State']
y=RAPESEEDAndMUSTARD['Cost of Cultivation (`/Hectare) A2+FL']
z=RAPESEEDAndMUSTARD['Cost of Cultivation (`/Hectare) C2']
a=RAPESEEDAndMUSTARD['Cost of Production (`/Quintal) C2']
b=RAPESEEDAndMUSTARD['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("RAPESEED And MUSTARD Production Variation", fontsize=16, fontweight='bold')
plt.show()

plt.figure(figsize=(10,8))
x=SUGARCANE['State']
y=SUGARCANE['Cost of Cultivation (`/Hectare) A2+FL']
z=SUGARCANE['Cost of Cultivation (`/Hectare) C2']
a=SUGARCANE['Cost of Production (`/Quintal) C2']
b=SUGARCANE['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("SUGARCANE Production Variation", fontsize=16, fontweight='bold')
plt.show()

plt.figure(figsize=(10,8))
x=WHEAT['State']
y=WHEAT['Cost of Cultivation (`/Hectare) A2+FL']
z=WHEAT['Cost of Cultivation (`/Hectare) C2']
a=WHEAT['Cost of Production (`/Quintal) C2']
b=WHEAT['Yield (Quintal/ Hectare) ']
plt.plot(x,y,label="Cost of Cultivation (`/Hectare) A2+FL")
plt.plot(x,z,label="Cost of Cultivation (`/Hectare) C2")
plt.plot(x,a,label="Cost of Production (`/Quintal) C2")
plt.plot(x,b,label="Yield (Quintal/ Hectare)")
plt.legend(loc="upper left")
plt.title("WHEAT Production Variation", fontsize=16, fontweight='bold')
plt.show()
import seaborn as sns
sns.pairplot(data=data1,kind="reg")
data2=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (2).csv')
data2.head()
data2.describe().T
Data2=data2.describe().T
dat=Data2.loc[:,"mean"]
dat.plot(kind='bar')
data2.head()
sns.pairplot(data=data2,kind="reg",y_vars=('Production 2006-07','Production 2007-08','Production 2008-09','Production 2009-10','Production 2010-11'),x_vars=('Area 2006-07','Area 2007-08','Area 2008-09','Area 2009-10','Area 2010-11'))
data3=pd.read_csv('../input/agricuture-crops-production-in-india/datafile (3).csv')
data3.groupby('Crop')
data4=pd.read_csv('../input/agricuture-crops-production-in-india/datafile.csv')
data4
data5=pd.read_csv('../input/agricuture-crops-production-in-india/produce.csv')
data5.T