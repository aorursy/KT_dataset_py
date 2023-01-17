# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
idata = pd.read_csv('/kaggle/input/2018-2010_import.csv')

edata = pd.read_csv('/kaggle/input/2018-2010_export.csv')

print(idata.head())

print(edata.head())
countrynames = idata.country.unique()

print(countrynames)
ivalue = idata[idata.country=='JAPAN']

evalue = edata[edata.country=='JAPAN']

#print(ivalue.head())

yeardata = ivalue.groupby('year').value.sum().reset_index()

yeardata1 = evalue.groupby('year').value.sum().reset_index()

print(yeardata)

plt.plot(yeardata.year,yeardata.value,label='Import')

plt.plot(yeardata1.year,yeardata1.value,label='Export')

plt.legend()

plt.show()
def plotexportimport(cname):

    ivalue = idata[idata.country==cname]

    evalue = edata[edata.country==cname]

    #print(ivalue.head())

    yeardata = ivalue.groupby('year').value.sum().reset_index()

    yeardata1 = evalue.groupby('year').value.sum().reset_index()

    #print(yeardata)

    plt.plot(yeardata.year,yeardata.value,label='Import')

    plt.plot(yeardata1.year,yeardata1.value,label='Export')

    plt.title(cname)

    plt.legend()

    plt.show()
for cname in countrynames:

    plotexportimport(cname)