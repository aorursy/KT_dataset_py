import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
sns.set(style="darkgrid")
register_matplotlib_converters()
# Input data files are available in the "../input/" directory.
# Import data
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# Drop unneded columns and parse dates
# Drop 'Province/State' data because the purpose is to compare between countries.
df = df.drop(columns=['SNo', 'Province/State', 'Last Update'])
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
# Group by selected countries: Spain, Italy, China, United Kingdom (UK)
gb_countries = df.groupby(by='Country/Region')
df_spain = gb_countries.get_group('Spain').reset_index(drop=True)
df_italy = gb_countries.get_group('Italy').reset_index(drop=True)
df_china = gb_countries.get_group('Mainland China').reset_index(drop=True)
df_uk = gb_countries.get_group('Germany').reset_index(drop=True)


# Flat cases of countries with data per province/state
# Among the selected countries, only China and the UK has data by province/state.
sum_cases_china = df_china.groupby('ObservationDate').sum()
df_china = df_china.drop(columns=['Confirmed', 'Deaths', 'Recovered']).merge(sum_cases_china, on='ObservationDate').drop_duplicates().reset_index(drop=True)
sum_cases_uk = df_uk.groupby('ObservationDate').sum()
df_uk = df_uk.drop(columns=['Confirmed', 'Deaths', 'Recovered']).merge(sum_cases_uk, on='ObservationDate').drop_duplicates().reset_index(drop=True)
df_uk.tail()
# Update last data [1]
# Recovered cases are published the next day, so added as the number of the previous day temporarily.

upd_spain = pd.DataFrame([['2020-04-01', 'Spain', 19980, 1002, 1588]], columns=df_spain.columns)
upd_spain['ObservationDate'] = pd.to_datetime(upd_spain['ObservationDate'])

upd_italy = pd.DataFrame([['2020-04-01', 'Italy', 41035, 3405, 4440]], columns=df_spain.columns)
upd_italy['ObservationDate'] = pd.to_datetime(upd_italy['ObservationDate'])

upd_china = pd.DataFrame([['2020-04-01', 'China', 80967, 3248, 71150]], columns=df_spain.columns)
upd_china['ObservationDate'] = pd.to_datetime(upd_china['ObservationDate'])

upd_uk = pd.DataFrame([['2020-04-01', 'Germany', 3269, 144, 65]], columns=df_spain.columns)
upd_uk['ObservationDate'] = pd.to_datetime(upd_uk['ObservationDate'])


df_spain = df_spain.append(upd_spain).reset_index(drop=True)
df_italy = df_italy.append(upd_italy).reset_index(drop=True)
df_china = df_china.append(upd_china).reset_index(drop=True)
df_uk = df_uk.append(upd_uk).reset_index(drop=True)
upd_uk.tail()
# Calculate ratios per population [2]
spain_pop = 46736776
italy_pop = 60550075
china_pop = 1433783686
# https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Bevoelkerungsstand/_inhalt.html
uk_pop    = 83100000

df_spain['Cases per Population Ratio'] = df_spain['Confirmed'] / spain_pop
df_italy['Cases per Population Ratio'] = df_italy['Confirmed'] / italy_pop
df_china['Cases per Population Ratio'] = df_china['Confirmed'] / china_pop
df_uk['Cases per Population Ratio'] = df_uk['Confirmed'] / uk_pop

df_spain['Deaths per Population Ratio'] = df_spain['Deaths'] / spain_pop
df_italy['Deaths per Population Ratio'] = df_italy['Deaths'] / italy_pop
df_china['Deaths per Population Ratio'] = df_china['Deaths'] / china_pop
df_uk['Deaths per Population Ratio'] = df_uk['Deaths'] / uk_pop

df_spain['Actives per Population Ratio'] = (df_spain['Confirmed'] - df_spain['Deaths'] - df_spain['Recovered']) / spain_pop
df_italy['Actives per Population Ratio'] = (df_italy['Confirmed'] - df_italy['Deaths'] - df_italy['Recovered']) / italy_pop
df_china['Actives per Population Ratio'] = (df_china['Confirmed'] - df_china['Deaths'] - df_china['Recovered']) / china_pop
df_uk['Actives per Population Ratio'] = (df_uk['Confirmed'] - df_uk['Deaths'] - df_uk['Recovered']) / uk_pop
# Add timeline of important events for each selected country
# This timeline is an **approximation** of what happened. [3][4]
# At this moment, there is no relevant events in the UK.
# The purpose is to figure out the pace of the measures rather than have an exact timeline of every detailed event.
events_spain = pd.DataFrame(columns=['ObservationDate', 'Event'],
                            data=np.array([['2020-03-09', 'Area Outbreak Close of schools'],
                                           ['2020-03-13', 'Close of commercials'],
                                          ['2020-03-15', 'Country Quarantine']]))
events_italy = pd.DataFrame(columns=['ObservationDate', 'Event'],
                            data=np.array([['2020-03-11', 'Close of commercials'],
                                          ['2020-03-09', 'Country Quarantine'],
                                          ['2020-03-08', 'Area Outbreak Quarantine'],
                                          ['2020-02-23', 'City Outbreak Close of schools']]))
events_china = pd.DataFrame(columns=['ObservationDate', 'Event'],
                            data=np.array([['2020-03-06', 'Travel ban to USA'],
                                           ['2020-02-16', 'Country Quarantine'],
                                           ['2020-01-29', 'Airlines start cancel flights'],
                                           ['2020-01-22', 'City Outbreak Quarantine'],
                                          ['2020-01-15', 'City Outbreak Close of schools']]))
events_uk = pd.DataFrame(columns=['ObservationDate', 'Event'],
                         data=np.array([['2020-03-16', 'Close of schools']]))

events_spain['ObservationDate'] = pd.to_datetime(events_spain['ObservationDate'])
events_italy['ObservationDate'] = pd.to_datetime(events_italy['ObservationDate'])
events_china['ObservationDate'] = pd.to_datetime(events_china['ObservationDate'])
events_uk['ObservationDate'] = pd.to_datetime(events_uk['ObservationDate'])

df_spain = df_spain.merge(events_spain, on='ObservationDate', how='left')
df_italy = df_italy.merge(events_italy, on='ObservationDate', how='left')
df_china = df_china.merge(events_china, on='ObservationDate', how='left')
df_uk = df_uk.merge(events_uk, on='ObservationDate', how='left')
def plot_comparison(x='ObservationDate', y='Confirmed', annotations=False):
    fig = plt.figure(figsize=(20,10))
    ax = sns.lineplot(x=x, y=y, data=df_spain, color="r", label='Spain')
    sns.lineplot(x=x, y=y, data=df_italy, ax=ax, color="g", label='Italy')
    sns.lineplot(x=x, y=y, data=df_china, ax=ax, color="y", label='China')
    sns.lineplot(x=x, y=y, data=df_uk, ax=ax, color="b", label='Germany')

    if annotations:
        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(
            arrowstyle = "->",
            connectionstyle = "angle,angleA=0,angleB=90,rad=10",
            color='black')

        dfs = [df_spain, df_italy, df_china, df_uk]
        for df in dfs:
            for row in df.iterrows():
                if isinstance(row[1][-1], str):
                    # TODO: Improve annotation locations to not clash between them
                    ax.annotate(row[1][-1], xy=(df[x].iloc[row[0]], df[y].iloc[row[0]]), xytext=(0, 50), textcoords='offset points', bbox=bbox, arrowprops=arrowprops)

        ax.axvline(np.datetime64('2020-03-13'), color='r', label='USA ban Schengen area')

    date_form = DateFormatter("%W")
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlabel('Week in the year calendar')
    ax.set_ylabel(y)
    plt.show()
plot_comparison(y='Confirmed', annotations=False)
plot_comparison(y='Deaths', annotations=False)
plot_comparison(y='Cases per Population Ratio', annotations=False)
plot_comparison(y='Cases per Population Ratio', annotations=True)
plot_comparison(y='Deaths per Population Ratio', annotations=False)
plot_comparison(y='Deaths per Population Ratio', annotations=True)
plot_comparison(y='Actives per Population Ratio', annotations=False)
plot_comparison(y='Actives per Population Ratio', annotations=True)