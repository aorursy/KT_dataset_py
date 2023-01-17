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
df = pd.read_excel("/kaggle/input/taxdata2/tax_data.xlsx")

df3= df.copy()

df2 = pd.read_excel("/kaggle/input/state-tax/tax_data.xlsx")

df.head()
items = [ 'Corp Tax Revenue (miilions)', 'Total Revenue',

       'Unemployment ', 'GDP']

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib
df3.head()
df3.groupby("State ").plot(x= 'Year', y= 'Unemployment ' )

fig, ax= plt.subplots()

styles = ['solid', 'dashed']

df3.groupby("State ").plot(x= 'Year', y= 'Unemployment ', ax= ax, ls= '-')

ax.legend(('Indiana',' Comparsion Cohort'))

ax.set_xlabel('Year')

ax.set_ylabel('Unemployment Rate')

ax.set_title('State Unemployement Rate')

plt.show()

plt.savefig('unemploy.png')
for item in items:

    if item == 'GDP':

        fig, axs= plt.subplots()    

        df3.groupby("State ").plot(x= 'Year', y= item , ax= axs, ls= '-')

        axs.legend(('Indiana',' Comparsion Cohort', 'Kansas', 'Ohio','North Carolina'),prop={'size': 6})

        axs.set_xlabel('Year')

        axs.set_ylabel(item)

        axs.axvline(x = 2013)

        axs.set_xlim([2010,2018])

        axs.set_ylim([40000,55000])

        plt.show()

        

    else:

        fig, axs= plt.subplots()    

        df3.groupby("State ").plot(x= 'Year', y= item , ax= axs, ls= '-')

        axs.legend(('Indiana',' Comparsion Cohort', 'Kansas', 'Ohio','North Carolina'),prop={'size': 6})

        axs.set_xlabel('Year')

        axs.axvline(x = 2013)

        axs.set_ylabel(item)

        plt.show()

     

  

    



for item in items:

    if item == 'GDP':

        fig, axs= plt.subplots()    

        df2.groupby("State ").plot(x= 'Year', y= item , ax= axs, ls= '-')

        axs.legend(('Indiana',' Comparsion Cohort'))

        axs.set_xlabel('Year')

        axs.set_ylabel(item)

        axs.axvline(x = 2013)

        axs.set_xlim([2010,2018])

        axs.set_ylim([40000,55000])

        plt.show()

        

    else:

        fig, axs= plt.subplots()    

        df2.groupby("State ").plot(x= 'Year', y= item , ax= axs, ls= '-')

        axs.legend(('Indiana',' Comparsion Cohort'))

        axs.set_xlabel('Year')

        axs.axvline(x = 2013)

        axs.set_ylabel(item)

        plt.show()

     