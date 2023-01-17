# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
path ='/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv'

data_emissions = pd.read_csv(path)

data_emissions.head()
data_emissions.info()



data_emissions.isnull().sum()

data_emissions.drop('Code', axis = 1, inplace=True)
data_emissions.rename(columns={'Annual CO₂ emissions (tonnes )':'CO2'}, inplace=True)
data_emissions.head()


def visualise_country(country):

    

    '''Creating a seperate dataframe'''

    data_emissions_vis = data_emissions[data_emissions['Entity'] == country]

    tot_yr = data_emissions_vis.Year.max() - data_emissions_vis.Year.min()

    tot_em = data_emissions_vis.CO2.sum()

    print(f"Total Co2 Emissions by {country} in {tot_yr} years: {'{:.2f}'.format(tot_em)} tonnes")

    

    '''Plot'''

    fig = sns.lineplot(data=data_emissions_vis, x="Year", y='CO2')

    plt.title('Co2 Emissions by ' + country + ' in '+str(tot_yr)+' years \n', fontsize=20)

    plt.ylabel('Co2 Emissions')

    
visualise_country('Spain')
total_emissions = data_emissions.groupby('Entity')['CO2'].sum()

total_emissions.sort_values(ascending=False)