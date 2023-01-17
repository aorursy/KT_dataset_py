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



pd.set_option('mode.chained_assignment', None)



# Any results you write to the current directory are saved as output.


startups = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding= 'unicode_escape')

startups.head()
startups.shape
startups.groupby("country_code").size().sort_values(ascending=False).head(10)
startups.groupby("country_code").size().sort_values(ascending=False).head(10).plot.pie()
canadian_startups = startups[startups.country_code == 'CAN']

canadian_startups.head()





canadian_startups.set_index('name').loc["AQUA PURE",'debt_financing']
canadian_startups.groupby(' market ')['debt_financing'].mean().sort_values(ascending=False).head(10)
canadian_startups = canadian_startups[ ['name',' market ','status',' funding_total_usd ','debt_financing']]

canadian_startups.head()

# data conversion was needed. column was converted from string to integer.

canadian_startups[' funding_total_usd '] = canadian_startups[' funding_total_usd '].str.replace(',', '')

canadian_startups['int_funding_total_usd'] =  pd.to_numeric(canadian_startups[' funding_total_usd '],errors='coerce')



canadian_startups['debt_funding_percentage'] = (canadian_startups['debt_financing'] / canadian_startups['int_funding_total_usd'])

canadian_startups.head()



canadian_startups.groupby(['status'])['debt_funding_percentage','int_funding_total_usd'].mean()