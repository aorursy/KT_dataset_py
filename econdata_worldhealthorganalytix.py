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
data=pd.read_csv('../input/who-dataset/WHO.csv')
data.head() #method with no params

data.tail(15) #method with no param
data.info()  #method with no param
data.columns #attributes of pandas dataframe
data.describe()
#Which country has the smallest percentage of the population over 60?
#Which country has the largest literacy rate?
data.Country.nunique()
data['Over60']
#euro12[euro12.Goals > 6]

data[data.Country=='Japan']
#United Arab Emirates (UAE)

data[data.Country=='United Arab Emirates ']

data[data.Country=='Sierra Leone']



data[data.Country=='Luxembourg']
data[data.Country=='Mali']
data[data.Country=='Cuba']
#Which region has the lowest average child mortality rate across all countries in that region?
data[data.ChildMortality==181.600000]
data.Region.unique()
data[data.Region=='Eastern Mediterranean'].describe()
data[data.Region=='Europe'].describe()
data[data.Region=='Africa'].describe()
data[data.Region=='Americas'].describe()
data[data.Region=='Western Pacific'].describe()
data[data.Region=='South-East Asia'].describe()