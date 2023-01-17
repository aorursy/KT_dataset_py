import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import date

from datetime import time

from datetime import datetime
# data source for the 8 stations in southeast michigan: http://mrcc.isws.illinois.edu/CLIMATE/

# it needs a login/account to access the data

# the data is uploaded here in csv format

precep=pd.read_csv('../input/precep.csv')

precep.head()
#total length of each data should be

print("Total length=", len (precep))

# using the pandas function describe we can analyze the summary stat of each column in the precep dataframe

precep.describe()

# looking at the count row, we can see some stations have missing data (i.e.length < 6209)
type(precep['Date'][1])

d=pd.to_datetime(precep['Date']) # changing date to timestamp

precep['date_new']=d

precep2=precep[['date_new','Genesee','Washtenaw','Macomb','Monroe','St_claire','Oakland','Livingston','Wayne','year','month']]

precep2.head(6)
# using fillna to fill gaps, but we have to set the limit; this works as long as we know the size of the gap.

#We can also set the limit to None so that gap of any size can be filled

precep_new=precep2.fillna(method='ffill', limit=None)

precep_new.head(5)



# visualizing the filled precipitation data for each county

nr=2

nc=4

fig= plt.figure(figsize=(10,8))#,sharex=True,sharey=True)

for i in range(nr):

    for j in range(nc):

        ax=plt.subplot2grid((nr,nc),(i, j))

        dp=j+i*nc

        ax.plot(precep_new.iloc[0:100,dp+1])

        ax.set_ylim(0,2)

        ax.legend(loc='upper right')

        ax.set_title('Filled')

fig.tight_layout()

plt.show()

    
#for i in len(precep_new['year']):

precep_inter=precep2

precep_inter=precep2.interpolate()

precep_inter.head(10)

# plotting interpolated, filled and original timeseries

nr=2

nc=4

fig2= plt.figure(figsize=(10,5))#,sharex=True,sharey=True)

for i in range(nr):

    for j in range(nc):

        ax=plt.subplot2grid((nr,nc),(i, j))

        dp=j+i*nc # this variable dp is used to locate the data column from the precipitation data

        line1=ax.plot(precep_inter.iloc[0:100,dp+1])  # we start from the second colum since the first coulumn is 'date_new'

        line2=ax.plot(precep_new.iloc[0:100,dp+1])

        line3=ax.plot(precep2.iloc[0:100,dp+1])

        ax.set_ylim(0,2)

        ax.legend(('Interpolated', 'Forward-filled', 'Original'),loc='upper right')

fig2.tight_layout()

plt.show()