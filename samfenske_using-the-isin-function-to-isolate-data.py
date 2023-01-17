# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image
Image('/kaggle/input/github/github.png')
url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/06-20-2020.csv'

covid=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/06-20-2020.csv')

covid.head()
#directory of all methods that may be called on a dataframe

dir(pd.DataFrame())
#search the entire dataframe for elements that contain the phrase 'US'



#covid is the dataframe we are searching

#'US' is the phrase we are searching for

    #'US' is in brackets because the .isin() method is bulit to accept a list of items, so even if you're searching for one

    #item, it still needs to be in brackets

covid.isin(['US']).head()
covid[covid.isin(['US'])].head()
covid['Country_Region'].isin(['US']).head()
covid[covid['Country_Region'].isin(['US'])].head()
covid.isin(['Alabama']).head()
covid[covid.isin(['Alabama'])].head()
covid['Province_State'].isin(['Alabama']).head()
covid[covid['Province_State'].isin(['Alabama'])].head()
covid.head(1)
pollution=pd.read_csv('/kaggle/input/uspollution/pollution_us_2000_2016.csv')

pollution.head()
#get data for state of missouri

pollution['State'].isin(['Missouri'])
pollution['State'].isin(['Missouri']).value_counts()
missouri=pollution[pollution['State'].isin(['Missouri'])]

missouri
#get data for Missouri and Illinois

pollution['State'].isin(['Missouri','Illinois']).value_counts()
#full data for both states

pollution[pollution['State'].isin(['Missouri','Illinois'])]
#when there a many columns, this is helpful to see what data you can look at

pollution.columns
pollution['State'].isin(['Missouri']).value_counts()
#direct comparison doesn't require .isin()

(pollution['CO Mean']>0.5).value_counts()
((pollution['CO Mean']>0.5) & pollution['State'].isin(['Missouri'])).value_counts()
pollution[(pollution['CO Mean']>0.5) & pollution['State'].isin(['Missouri'])].head()
mizzou=pollution['State'].isin(['Missouri'])

threshold=(pollution['CO Mean']>0.5)



#same dataframe as above, just utilizing saved variables

pollution[mizzou & threshold].head()