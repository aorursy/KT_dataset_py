import numpy as np

import pandas as pd

import datetime

from bokeh.charts import Scatter, Bar, show, output_notebook

output_notebook()

data = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt',sep=',')

data.sample()
# Return a bunch of tuples with the Zodiac and its Start/End Dates

def chinese_zodaics():

    start_date = pd.to_datetime("2/2/1908")

    end_date = pd.to_datetime("7/1/2009")

    animals = ['Monkey', 'Rooster', 'Dog', 'Pig', 'Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake', 'Horse', 'Goat']

    zodiacs = []

    while start_date < end_date:

        for a in animals:    

            year_start = start_date

            year_end = year_start + pd.DateOffset(days=365)

            z = (a, start_date, year_end)

            zodiacs.append(z)

            start_date = year_end

    return zodiacs 



zodiacs = chinese_zodaics()



# Apply the zodiacs to the accident dates

def match_zodiac(date):

    for z in zodiacs: 

        animal, start, end, = z[0], z[1], z[2]

        if start <= date <= end:

            return animal

        

data.Date = pd.to_datetime(data.Date)

data['Zodiac'] = data.Date.apply(match_zodiac)

data['Year'] = pd.DatetimeIndex(data['Date']).year

data = data[['Zodiac', 'Year', 'Fatalities', 'Aboard']].dropna()

data = data[data.Fatalities > 1]

data.sample(5)

data.describe().astype(int)
p = Scatter(data, x='Fatalities', y='Zodiac', marker='Zodiac', color='Zodiac',

            title="Fatalities by Zodiac", legend=None,

            xlabel="Fatalities", ylabel="Zodiac")





show(p)
# Put key stats into a DataFrame

def zodiac_data(data):

    idx=['Total_Accidents', 'Total_Deaths', 'Mean_Deaths', 'Death_Rate', 'Survival_Rate', 'Deadliest_Accident']

    df = pd.DataFrame()

    for z in data.Zodiac.unique(): 

        zodiac = data[data.Zodiac == z]

        f = zodiac.Fatalities.dropna()

        a = zodiac.Aboard

        total_accidents = f.count()

        total_deaths = f.sum()

        mean_deaths = f.mean()

        death_rate = total_deaths / a.sum()

        survival_rate = 1 - death_rate

        deadliest = f.max()

        df[z] = [total_accidents, total_deaths, mean_deaths, death_rate, survival_rate, deadliest]

    df.index = idx

    df = df.round(2).T

    return df



zodiac_comparison = zodiac_data(data)

zodiac_comparison
zodiac_comparison.describe().round(2)
p = Bar(data, label='Zodiac', values='Fatalities', agg='mean', stack='Zodiac',

        title="Average Annual Deaths by Zodiac", legend='top_right')







show(p)