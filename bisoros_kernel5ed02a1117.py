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
import pandas as pd 



data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data = data[data['Country/Region']=='UK'][['ObservationDate', 'Confirmed', 'Deaths', 'Recovered']]
from datetime import date

from datetime import datetime



def toDays(date):

    f_date = datetime(2020, 1, 31)

    l_date = datetime.strptime(date, '%m/%d/%Y')

    delta = l_date - f_date

    return delta.days
data['dayNo'] = data['ObservationDate'].apply(lambda row: toDays(row))
data
total = data[['Confirmed', 'Deaths', 'Recovered']].cumsum()

total = total.rename(columns={

    'Confirmed' : 'TotalConfirmed',

    'Deaths' : 'TotalDeaths',

    'Recovered' : 'TotalRecovered'

})

total
data = data.join(total)
data