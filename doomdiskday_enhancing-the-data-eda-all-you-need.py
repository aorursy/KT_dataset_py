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
!pip install COVID19Py --upgrade

!pip install cufflinks --upgrade

!pip install wget



import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import copy

import COVID19Py

import matplotlib.pyplot as plt

import wget



def print_full(x):

    pd.set_option('display.max_rows', len(x))

    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', 2000)

    pd.set_option('display.float_format', '{:20,.2f}'.format)

    pd.set_option('display.max_colwidth', -1)

    print(x)

    pd.reset_option('display.max_rows')

    pd.reset_option('display.max_columns')

    pd.reset_option('display.width')

    pd.reset_option('display.float_format')

    pd.reset_option('display.max_colwidth')

    

def read_train():

    train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

    return train_df
def get_recovered_by_country():

    

    confirmed = {}

    

    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    filename = wget.download(url)



    recv_df = pd.read_csv(filename)

    dates = recv_df.columns[4:]



    recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')

    

    

    recv_df_long["Date"] = pd.to_datetime(recv_df_long.Date)

    

    for i, row in recv_df_long.iterrows():

        country = row['Country/Region']

        recovered = row['Recovered']

        date = row['Date']

        if country not in confirmed:

            confirmed[country] = {}

        

        _month = date.month if len(str(date.month)) != 1 else "0{}".format(date.month)

        _str_date = "{}-{}-{}".format(date.year, _month, date.day)

        confirmed[country][_str_date] = recovered

        

    

    return confirmed



def add_recovered_cases(recovered_cases, df):

    def add_recovered(row):

        return recovered_cases.get(row[2], {}).get(row[5],0)

    

    df['Recovered'] = df.apply(lambda x: add_recovered(x), axis=1)

    return df







df = read_train()



recovered_cases = get_recovered_by_country()



df = add_recovered_cases(recovered_cases, df)



countries = df["Country/Region"].unique()

df_by_date = df.groupby("Date").sum()

last_day = df_by_date.iloc[-1]

pre_last_day = df_by_date.iloc[-2]

total_infected = last_day["ConfirmedCases"]

total_deaths = last_day["Fatalities"]

total_recoveries = pre_last_day["Recovered"]



infected_timeline = df_by_date["ConfirmedCases"]

death_timeline = df_by_date["Fatalities"]

recovery_timeline = df_by_date["Recovered"][:-1]



print("[!] GENERAL STATISTICS")

print("[-] Number of infected countries: {}".format(len(countries)))

print("[-] Total number of infected: {}".format(total_infected))

infected_timeline.iplot()

print("[-] Total number of deaths: {}".format(total_deaths))

death_timeline.iplot()

print("[-] Total number of recoveries: {}".format(total_recoveries))

recovery_timeline.iplot()





def add_insights(_df):

    # Here we add some insight:

    # - fatal ratio and recovery ratio

    # - a balanced fatal ratio, which takes into account the number of confirmed cases.

    # - a balanced recovery ratio, as above.

    

    #_df = _df.groupby(["Country/Region"]).sum()



    

    _df = _df[["Country/Region","ConfirmedCases", "Fatalities", "Recovered"]]

    _df['FatalRatio'] = (_df['Fatalities'] / _df['ConfirmedCases']) *100

    

    _df['RecoveryRatio'] = (_df['Recovered'] / _df['ConfirmedCases']) *100



    _df.replace(np.inf, 0, inplace=True)

    _df.replace(np.nan, 0, inplace=True)

    return _df



def pie_column(df, column="FatalRatio", top=5):

    

    _tmp = df[['Country/Region', column]][:top]

    

    #plot = _tmp.plot.pie(y=column, figsize=(5, 5))

    values = [x[1] for i,x in _tmp.iterrows()]

    labels = [str(x) for x in _tmp['Country/Region']]



    fig1, ax1 = plt.subplots()

    ax1.pie(values, autopct='%1.1f%%', labels=labels, shadow=True, startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle

    ax1.axis('equal')  

    plt.tight_layout()

    plt.show()

    

def analyze_fatality(_df, top=5):

    _df = _df.sort_values("FatalRatio", ascending=False)

    print("[-] Cases by country, sorted by fatality ratio")

    print_full(_df.head(top))

    

    pie_column(_df, column="FatalRatio", top=top)

    

    _df = _df.sort_values("Fatalities", ascending=False)

    print("[-] Cases by country, sorted fatalities")

    print_full(_df.head(top))

    

    pie_column(_df, column="Fatalities")

    

def analyze_recovery(_df, top=5):

    _df = _df.sort_values("RecoveryRatio", ascending=False)

    print("[-] Cases by country, sorted by recovery ratio")

    print_full(_df.head(top))

    

    pie_column(_df, column="RecoveryRatio", top=top)

    

    _df = _df.sort_values("Recovered", ascending=False)

    print("[-] Cases by country, sorted by number of recevered")

    print_full(_df.head(top))

    

    pie_column(_df, column="Recovered")



def create_country_df(df):

    country_df = df.groupby(["Country/Region", "Date"]).sum()

    last_day_df = []

    

    all_countries = list(df['Country/Region'].unique())

    

    for country in all_countries:

        last_infos = country_df.loc[country].iloc[-1].to_dict()

        last_infos['Country/Region'] = country

        last_day_df.append(last_infos)

        

    _df = pd.DataFrame(last_day_df)

    return _df

    

df = read_train()



recovered_cases = get_recovered_by_country()



df = add_recovered_cases(recovered_cases, df)



country_df = create_country_df(df)





insight_country_df = add_insights(country_df)



analyze_fatality(insight_country_df)

analyze_recovery(insight_country_df, top=5)