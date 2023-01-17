# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pwd
df = pd.read_csv('../input/avocado-prices/avocado.csv')

#do this and then we will parse the dates as dates and set the right column as index
pd.__version__
pd.options.display.max_rows = 5

#better set this so that the output doesn't take the whole page
df.head()
%matplotlib inline

#before plotting
df.plot()

#not that useful really .. but will get there
df.head()

#let's see the header again ... So Date should be the index

df['Date'] = pd.to_datetime(df.Date)
df.set_index('Date')

#see what happens.. it shows the output
df.head()

#as you can re-check the header .. Date is not the index becasue we didn't use the inplace
df.set_index('Date',inplace = True)
#let us see the header again 

df.head()
#now let us plot and see 

df.plot()
#somewhat better plot than previous but let ius plot for one column only

df.AveragePrice.plot()
#looks good for one column but not that useful

albany_df = df.copy()[df.region == 'Albany']
albany_df

#printout the slice that we have got
#let us see the plot of average prices

albany_df.AveragePrice.plot()
#better plot than average prices for all regions  but still can be improved

albany_df['Total Bags'].plot()
#let us look at the index now

albany_df.index
#let us sort it first

albany_df.sort_index()

#remember it gives out a view only.. so inplace
albany_df.sort_index(inplace=True)
albany_df.head()

#looks good
#let us plot again - same problem of lot of data points

df.AveragePrice.plot()
#how many rows
len(albany_df)
#let us use 25 rolling mean so that 14 points are generated

338/25
albany_df.AveragePrice.rolling(25).mean()

#it will be NAN for 24 rows and then values appear
albany_df.AveragePrice.rolling(25).mean().head(26)
#unable to see all the ouput so let us change the settings

pd.options.display.max_rows = 50
albany_df.AveragePrice.rolling(25).mean().head(26)
#let us plot it 

albany_df.AveragePrice.rolling(25).mean().plot()
#so what are the unique regions

df.region
df.region.tolist()
set(df.region.tolist())
len(set(df.region.tolist()))
df.region.nunique()
df.region.unique()
type(df.region.unique())

#gives out a NumPY array with charaters
unique = np.array(list(set(df.region.tolist())))

#lot of long process to convert the types -- but shows power of Python !
type(unique)
unique
'Boston' in unique

#how to check the if a name is in the region list
#we can sort NumPy array very easily!

df.region.unique().sort()
df.region.unique()

#check that order is already sorted