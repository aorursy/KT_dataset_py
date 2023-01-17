#Question :
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/ '):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot
sf = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')
af = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')
af.info()
sf.info()
sf['InvoiceDate']=pd.to_datetime(sf['InvoiceDate'],format='%m/%d/%Y')
sf2=sf.set_index("InvoiceDate").groupby(pd.Grouper(freq='M'))['Quantity'].sum().reset_index(name='total Quantity per bulan')
sf.duplicated().sum()
sf2.plot(kind='bar', x='InvoiceDate')
sf2.sort_values('total Quantity per bulan',ascending=False)
#Insight :

#Total pengeluaran marketing terbesar untuk offline terjadi pada bulan November 2017, sebesar 96600. 

#Pada bulan yang sama, total penjualan offline sebesar 83381. 

#

#