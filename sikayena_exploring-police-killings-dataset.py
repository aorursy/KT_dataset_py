# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import other libraries

import matplotlib.pyplot as plt

import seaborn as sb

sb.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# let's just sneak a peek at the data

!head -n2 ../input/police_killings2015.csv



#seems to be printing more data than we expected. Is there some issue with the EOLs in the file?
polkills = pd.read_csv('../input/police_killings2015.csv');

polhills.head()

# this throws an encoding error. apparently this CSV file is not utf-8 encoded.
# let's see if we can determine the encoding

!file -i ../input/police_killings2015.csv
# hooray! it seems this is uses iso-8859-1 encoding not utf-8.

# perhaps pd.read_csv will work now with a specified encoding... here goes

polkills = pd.read_csv('../input/police_killings2015.csv', encoding='iso-8859-1');
# awesome. success! let's see what the dataframe contains now

polkills.head()

# more awesome.

print(polkills.shape)
# what features are in the data and what are the datatypes

polkills.info()
# check for any nulls

print(polkills.isnull().any())
# let's get a count of null/nan entries in the 

#a = np.array([(c, polkills[c].isnull().sum()) for c in polkills.columns ])

#print(a)

# most nulls occur in county_bucket column. what do the entries there look like

#print('')

nan_cols = ['streetaddress','h_income','comp_income','county_bucket','nat_bucket','urate','college']



#polkills.iloc([pd.isnull(polkills).sum() > 0])

for k in nan_cols:

    print('{}:'.format(k))

    print(polkills[polkills[k].isnull()])
polkills.ix[379,:]

# state_fp data here https://en.wikipedia.org/wiki/Federal_Information_Processing_Standard_state_code 

# FIPS codes https://www.census.gov/geo/reference/codes/cou.html

# polkills.describe()
#country_codes= pd.read_excel('https://www.census.gov/geo/reference/codes/cou.html')