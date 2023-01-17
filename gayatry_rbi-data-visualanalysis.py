#importing basic libraries

import numpy as np

import pandas as pd

from pandas import Series,DataFrame



import xlrd



import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
#importing data

data = pd.read_excel('/kaggle/input/rbi-data-columnaligned/RBI_Data.xlsx',header=[0,1])

data.head()
data = data.round(2)

data.head()
data = data.set_index(('Components and Sources','Date'))

data.head()
data.info()
data = data.replace('-',0)
data.info()
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))



ax1.plot(data['Components']['Currency in circulation -Total '])

ax2.plot(data['Components']['Other deposits with RBI '])

ax3.plot(data['Components']['Bankers deposits with RBI '])



ax1.set_title('Currency in circulation -Total')

ax2.set_title('Other deposits with RBI')

ax3.set_title('Bankers deposits with RBI')
data['Reserve Money']['Reserve Money (Liabilities/Components) '].plot()
fig,ax = plt.subplots(4,2,figsize=(20,10))

fig.tight_layout(pad=3.0)



ax[0,0].plot(data['Sources']["RBI's Claims on - Government (net)"])

ax[0,0].set_title("RBI's Claims on - Government (net)")

ax[0,1].plot(data['Sources']["RBI's Claims on - Central Govt"])

ax[0,1].set_title("RBI's Claims on - Central Govt")

ax[1,0].plot(data['Sources']["RBI's Claims on Banks & Commercial sector"])

ax[1,0].set_title("RBI's Claims on Banks & Commercial sector")

ax[1,1].plot(data['Sources']["RBI's Claims on Banks (Including NABARD)"])

ax[1,1].set_title("RBI's Claims on Banks (Including NABARD)")

ax[2,0].plot(data['Sources']["RBI's claims on Commercial sector (Excluding NABARD) "])

ax[2,0].set_title("RBI's claims on Commercial sector (Excluding NABARD)")

ax[2,1].plot(data['Sources']["Net foreign exchange assets of RBI "])

ax[2,1].set_title("Net foreign exchange assets of RBI ")

ax[3,0].plot(data['Sources']["Govt't currency liabilities to the public "])

ax[3,0].set_title("Govt't currency liabilities to the public ")

ax[3,1].plot(data['Sources']["Net non-monetary liabilities of RBI "])

ax[3,1].set_title("Net non-monetary liabilities of RBI ")