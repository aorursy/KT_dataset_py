# Generic Libraries

import numpy as np

import pandas as pd



# Visualisation Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import re



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-darkgrid')

pd.set_option('display.max_columns', 50)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.1f}'.format
url = '../input/crop-yield-per-state-and-rainfall-data-of-india/out.csv'



data = pd.read_csv(url, header='infer')
data.shape
data.isna().sum()
data.info()
data.head()
data = data.drop('Unnamed: 0', axis=1)
#Creating a data backup

data_backup = data.copy()
APR_df = data[['Area','Production','Rainfall']]



corr = APR_df.corr()

plt.figure(figsize=(8, 8))

g = sns.heatmap(corr, annot=True, cmap = 'PuBuGn_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})

g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')

g.set_title("Correlation between Area, Production & Rainfall", fontsize=14)

#bottom, top = g.get_ylim()

#g.set_ylim(bottom + 0.5, top + 0.5)

plt.show()
data.head()
#Creating a subset of the original data for visualisation

sub_data = data.drop(data[data.Season == 'Whole Year '].index)
# Create a function that returns Bar Graphs for Production & Rainfall in States per seasons

def seasonal_view(state):

    """

    Creating 2 seperate pivot-tables with mean Production & Rainfall

    """

    ptable_prod = sub_data[(sub_data['State'] == state)].pivot_table(values='Production',index='Year', columns='Season', aggfunc= 'mean', fill_value= 0.0)

    ptable_rain = sub_data[(sub_data['State'] == state)].pivot_table(values='Rainfall',index='Year', columns='Season', aggfunc= 'mean', fill_value= 0.0)





    fig = plt.figure()

    plt.subplots_adjust(hspace = 5)

    sns.set_palette('deep')

    

    """

    Draw a Line Graph on First subplot. - Rainfall

    """

    

    year_labels = ptable_rain.index.tolist()

    season_labels = ptable_rain.columns.tolist()

    

    ax1 = ptable_rain.plot(kind='line', figsize=(15,6))

    # Add some text for labels, title and custom x-axis tick labels, etc.

    plt.title(f'{state.capitalize()} Seasonal Annual (mean) Rainfall', fontsize=16)

    plt.ylabel("Rainfall", fontsize=13)

    plt.legend(prop={'size':10}, loc='best',bbox_to_anchor=(0.4, 0., 0.75, 0.5) )

    





    """

    Draw a Line Graph on First subplot. - Production

    """

    

    ax2 = ptable_prod.plot(kind='line',figsize=(15,6))

    plt.title(f'{state.capitalize()} Seasonal Annual (mean) Production', fontsize=16)

    plt.ylabel("Production", fontsize=13)

    plt.legend(prop={'size':10}, loc='best',bbox_to_anchor=(0.4, 0., 0.75, 0.5))



    fig.tight_layout()

    plt.show()
seasonal_view('Bihar')
seasonal_view('Punjab')
seasonal_view('Kerala')
seasonal_view('Jharkhand')
seasonal_view('Uttarakhand')
seasonal_view('Odisha')
seasonal_view('Chhattisgarh')