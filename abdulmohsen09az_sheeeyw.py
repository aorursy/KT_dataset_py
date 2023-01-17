# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
country_data = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")

data = pd.read_csv("../input/GlobalTemperatures.csv")
country_data['dt']= pd.to_datetime(country_data['dt'])

country_data['year']= country_data['dt'].map(lambda x:x.year)

country_data['month']= country_data['dt'].map(lambda x:x.month)
def get_season(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'autumn'

    else:

        return 'winter'

country_data['season']= country_data['month'].apply(get_season)
saudi = country_data[country_data['Country']=='Saudi Arabia']

saudi.head()
sns.lmplot(x="year", y="AverageTemperature", hue="season", data=saudi)