# Importing Python Libraries



import numpy as np

import pandas as pd

import datetime



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation



from IPython.display import HTML
import matplotlib

matplotlib.use("Agg")
# Importing dataset and seting "Last Update" and "ObservationDate" as date by parse_dates



dataset = (pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',

                       parse_dates = ['Last Update', 'ObservationDate'])

                       .sort_values(by='Last Update', ascending = False))

dataset.head(10)
# Visualize empty values in dataset

dataset.isna().sum()
#Rename Mainland China to China

dataset['Country/Region'].replace('Mainland China', 'China', inplace = True)



#Filling empty provinces with "NA"

dataset['Province/State'].fillna('NA', inplace = True)
list_infected_countries = pd.DataFrame(data = dataset['Country/Region'].sort_values().unique(), columns = {'Country/Region'})



num_infected_countries = len(list(dataset['Country/Region'].sort_values().unique()))



print("Actually there's %d countries infected by Corona Virus in the World \n \n" %

      len(list(dataset['Country/Region'].sort_values().unique())))



list_infected_countries
# Last observation date in dataset

last_data_day = dataset['ObservationDate'].max()



# Filtering the dataset with the selected date

df = dataset[dataset['ObservationDate'].eq(last_data_day)]



# Creating a dataset grouped by countries and sortened by confirmed cases

df_group = pd.DataFrame(data = (df.groupby(['Country/Region'], as_index = False)

      .sum()

      .sort_values(by='Confirmed', ascending=False)

      .head(10)

      .reset_index(drop=True)))



# Removing 'SNo' column

df_group.drop(columns = ['SNo'], inplace = True)



df_group
fig, ax = plt.subplots(figsize=(15, 8))

df_group = df_group[::-1]

ax.barh(df_group['Country/Region'], df_group['Confirmed'])
def draw_barchart(day):

    

    #Creating Top 10 Confirmed Dataset

    

    df = dataset[dataset['ObservationDate'].eq(day)]

    

    df_group = (df.groupby(['Country/Region'], as_index = False)

          .sum()

          .sort_values(by='Confirmed', ascending=False)

          .head(10)

          .reset_index(drop=True))



    df_group.drop(columns = ['SNo'], inplace = True)

    

    #Creating Bar Chart

    ax.clear()

    df_group = df_group[::-1]

    ax.barh(df_group['Country/Region'], df_group['Confirmed'])

    

    dx = df_group['Confirmed'].max() / 1000

    

    #Format Bar Chart

    for i, (value, name) in enumerate(zip(df_group['Confirmed'], df_group['Country/Region'])):

        

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='center')

        #ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')

    

    ax.text(1, 0.4, day.strftime("%d/%m/%Y"), transform=ax.transAxes, color='#777777', size=30, ha='right', weight=600)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Confirmed Corona Virus cases in the world',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, 'by @joaocampista', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)
fig, ax = plt.subplots(figsize=(15, 8))

draw_barchart(dataset['ObservationDate'].max())
fig, ax = plt.subplots(figsize=(15, 8))



day_zero = dataset['ObservationDate'].min()

day_target = dataset['ObservationDate'].max()

dates = list(pd.date_range(day_zero, day_target))



animator = animation.FuncAnimation(fig, draw_barchart, frames=dates) 



HTML(animator.to_jshtml())
dataset_w_c = dataset[~dataset['Country/Region'].eq('China')].sort_values(by='Confirmed', ascending = False)

dataset_w_c.head(10)
def draw_barchart_w_c(day):

    

    #Creating Top 10 Confirmed Dataset

    

    df = dataset_w_c[dataset_w_c['ObservationDate'].eq(day)]

    

    df = (df.groupby(['Country/Region'], as_index = False)

          .sum()

          .sort_values(by='Confirmed', ascending=False)

          .head(10)

          .reset_index(drop=True))



    df.drop(columns = ['SNo'], inplace = True)

    

    #Creating Bar Chart

    ax.clear()

    df = df[::-1]

    ax.barh(df['Country/Region'], df['Confirmed'])

    

    dx = df['Confirmed'].max() / 1000

    

    #Format Bar Chart

    for i, (value, name) in enumerate(zip(df['Confirmed'], df['Country/Region'])):

        

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='center')

        #ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')

    

    ax.text(1, 0.4, day.strftime("%d/%m/%Y"), transform=ax.transAxes, color='#777777', size=30, ha='right', weight=600)

    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Confirmed Corona Virus cases outside China',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, 'by @joaocampista', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

    



fig, ax = plt.subplots(figsize=(15, 8))

draw_barchart_w_c(dataset['ObservationDate'].max())
fig, ax = plt.subplots(figsize=(15, 8))



day_zero = dataset_w_c['ObservationDate'].min()

day_target = dataset_w_c['ObservationDate'].max()

dates = list(pd.date_range(day_zero, day_target))



animator = animation.FuncAnimation(fig, draw_barchart_w_c, frames=dates)



HTML(animator.to_jshtml())