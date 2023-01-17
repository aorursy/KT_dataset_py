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
import datetime

from datetime import date



today = date.today()



y = today.strftime("%Y")

m = today.strftime("%B")

d = today.strftime("%d")



yesterday = today - datetime.timedelta(days = 1)

y_y = yesterday.strftime("%Y")

y_m = yesterday.strftime("%B")

y_d = yesterday.strftime("%d")



#print(d, m)

print(y_d, y_m)
#todays date

date_url = d +'-'+ m + '-' + y



#yesterday date

#date_url = y_d +'-'+ y_m + '-' + y

url = 'https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2020/04/COVID-19-total-announced-deaths-' + date_url + '.xlsx'

url
link = url

data_region = pd.read_excel(link, 'COVID19 total deaths by region', skiprows=15)

data_region = data_region.dropna(how='all', axis='columns')

data_region = data_region[pd.notnull(data_region['NHS England Region'])]

data_region = data_region.reset_index(drop=True)



data_trust = pd.read_excel(link, 'COVID19 total deaths by trust', skiprows=15)

data_trust = data_trust.dropna(how='all', axis='columns')

data_trust = data_trust[pd.notnull(data_trust['NHS England Region'])]

data_trust = data_trust.reset_index(drop=True)





data_age = pd.read_excel(link, 'COVID19 total deaths by age', skiprows=15)

data_age = data_age.dropna(how='all', axis='columns')

data_age = data_age[pd.notnull(data_age['Age group'])]

data_age = data_age.reset_index(drop=True)

#data_region.head(10)

#data_trust.head(30)

#data_age.head(10)
#Create output file



data_region.to_csv('Fatalities_by_region_uk.csv', index = False)

data_trust.to_csv('Fatalities_by_trust_uk.csv', index = False)

data_age.to_csv('Fatalities_by_age_uk.csv', index = False)