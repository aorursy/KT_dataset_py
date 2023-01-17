import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

us_state_demo = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/state/detail/SCPRC-EST2019-18+POP-RES.csv")
us_state_demo.head()
training_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv",
                           low_memory=True)
country_data = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')
country_data.head(2)
country_data[country_data['Province.State'] == "California"]
training_data.head(2)
califonia_data = training_data[training_data.Province_State == "California"].tail()
califonia_data.index = califonia_data.Date
califonia_data.tail(2)
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.plot(
    califonia_data.loc['2020-03-15':'2020-04-2','ConfirmedCases'],
    marker='o',
    linestyle='-'
)
ax.set_ylabel("Confirmed Cases")
ax.plot(
    califonia_data.loc['2020-03-15':'2020-04-2','Fatalities'],
    marker='^',
    linestyle='-'
)
#ax.xaxis.set_major_formatter(
#    mdates.DateFormatter('%b %d')
#)
california_test_data = pd.read_html("https://covidtracking.com/data/state/california#historical")[1]
california_test_data.tail()
datetime.strptime(california_test_data.Date[0],'%d %b %Y %a')
def format_date(date_string : str) -> datetime:
    """
        Format string to datetime object
        :param date_string: date string
        :returns date_obj: parsed date as datetime object
    """
    try:
        date_obj = datetime.strptime(
            date_string,
            '%d %b %Y %a'
        )
        return date_obj
    except:
        date_obj = datetime.strptime(
            date_string,
            '%Y-%m-%d'
        )
        return date_obj

california_test_data.Date  = pd.to_datetime(california_test_data.Date)
california_test_data.Date[0]
us_test_data = pd.read_html("https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/testing-in-us.html")[0]
us_test_data.tail()

state_wise_data = pd.read_html("https://docs.google.com/spreadsheets/u/2/d/e/2PACX-1vRwAqp96T9sYYq2-i7Tj0pvTf6XVHjDSMIKBdZHXiCGGdNC0ypEU9NbngS8mxea55JuCFuua1MUeOj5/pubhtml#")[1]
state_wise_data.head(5)