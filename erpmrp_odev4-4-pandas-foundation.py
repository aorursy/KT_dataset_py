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
datas = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')



datas.head()
# cerate from scratch, dictionary:

city = ["Istanbul", "Ankara"]

plate = ["34", "06"]

list_col = [city, plate]



list_label = ["SEHIR:", "PLAKA"]



# zipleyelim, iki listeyi

zipped = list(zip(list_label, list_col))

print(zipped)
# zipped to-> dic:

ddic = dict(zipped)



# dic to->dataframe:

dfrm = pd.DataFrame(ddic)

dfrm
# yeni kolon ekle, diziden değer ata:

dfrm["BOLGE"] = ["marmara", "anadolu"]

dfrm
# yeni kolon ekle, tek değer ata

dfrm["NUFUS"] = 999

dfrm
# let's retun to, get data set

datas.head()
import matplotlib.pyplot as plt
# plot all data



datas1 = datas.loc[:,["Unit price","Tax 5%", "Total"] ]

datas1.plot()

plt.show()
# subplots

datas1.plot(subplots=True)

plt.show()
# scatter plot

datas1.plot(kind="scatter", x="Unit price", y="Total")

plt.show()
# histogram plot:

datas1.plot(kind="hist", y="Unit price", bins=50, range=(0,250), normed=True)

plt.show()
# histogram subplot, noncumul ve cumul.ile



fig,axes= plt.subplots(nrows=2, ncols=1)

datas1.plot(kind="hist", y="Unit price", bins=50, range=(0,250), normed=True, ax=axes[1], cumulative=True)

plt.show()
datas.describe()
datas.head()
datas2 = datas.head()

datas2
# add date column with a list, lste stringdir.

datelist = ["1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"]



# string liste-> datetime yapalım:

datetimeobj = pd.to_datetime(datelist)



# kolonu ekleyelim:

datas2["mydate"] = datetimeobj



datas2
# dataframe index ini, mydate yapalım:

datas2 = datas2.set_index("mydate")

datas2
datas2.head()
# belli bir index li kaydı çekelim,slice yapalım:

print(datas2.loc["1993-03-16"])
# belli zaman aralığındaki index li, kaydı çekelim,slice yapalım:

print(datas2.loc["1992-03-10":"1993-03-16"])
datas2.head()
# yıl:A a göre, ort. resample yap:

datas2.resample("A").mean()



#resample periyodundaki(YIL) son tarihleri alır, yanlarına ort. değerleri yazar.
# aylara göre resmple:

datas2.resample("M").mean()



# 

#resample periyodundaki(AY) son tarihleri alır, yanlarına ort. değerleri yazar.
# lineer interpolate, NaN values:

datas2.resample("M").mean().interpolate("linear")