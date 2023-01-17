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



# Any results you write to the current directory are saved as output.
data_path = '../input/ncc_consolidated.csv'

ncc = pd.read_csv(data_path)

ncc
ncc.head(3)
ncc.loc[0:10]
ncc.dtypes



# Drop unnecessary columns

ncc_data = ncc.copy().loc[:, (ncc.columns != 'ID') & (ncc.columns != 'Contract Description') &

           (ncc.columns != 'Comments') & (ncc.columns != 'Additional Comments')  

            & (ncc.columns != 'Column 13')]

ncc_data
ncc_data.dtypes
import datetime

ncc_data['Date'] = pd.to_datetime(ncc_data.Date)

ncc_data.dtypes
ncc_data.Date.dt.dayofyear.head()
ncc_data.columns = ncc_data.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
ncc_data
ncc_data['Dollar_Amount'] = ncc_data.Dollar_Amount.astype(int)

ncc_data['Jamaican_Equivalent'] = ncc_data.Jamaican_Equivalent.astype(int)
ncc_data.dtypes
#top 10 most costly contracts between 2010 - 2013

dateN1 = pd.to_datetime('1/1/2010')

date2010 = ncc_data.loc[ncc_data.Date > dateN1, :]

Finaldata = date2010.nlargest(10,'Dollar_Amount')

Finaldata



Finaldata.set_index('Date',inplace=True)

#plot data

fig, ax = plt.subplots(figsize=(15,7))

plt.ylabel('Cost per Billion',fontsize='12')

Finaldata.plot(kind='bar',ax=ax,title='Top 10 Costly Contracts: 2010 - 2013')
dateN2 = pd.to_datetime('1/1/2000')

date2000 = ncc_data.loc[ncc_data.Date > dateN2, :]



# Select all duplicate rows based on one column

duplicateRowsDF = date2000[date2000.duplicated(['Government_Agency'],keep=False)]

 

duplicateRowsDF
# number of times an agency was awarded a contract from 2000

ch = pd.value_counts(duplicateRowsDF['Government_Agency'].values, sort=False).nlargest(10)



ch.name = 'Number of Contracts Awarded since 2000'

pd.DataFrame(ch) # print data frame using jupyter
ch.max()
plotfig = pd.value_counts(duplicateRowsDF['Government_Agency'].values, sort=False).nlargest(10).plot(kind="bar",figsize=(12,6))

plt.ylabel('Amount of Contracts')

plt.xlabel('Agency Awarded')

plt.title('Most Awarded Contracts Since 2000')