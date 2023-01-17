# Generic Libraries

import numpy as np

import pandas as pd



# Visualisation Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import warnings

from matplotlib import cm



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")



from tabulate import tabulate

#Data Load

url = '../input/movie-data/imdb_movie_data.csv'

data = pd.read_csv(url, header='infer')
data = data.drop(columns=['Rank','Description','Actors'], axis=1)
data.head()
#Checking for Null Values

data.isna().sum()
#Dropping Records with missing Revenue

data = data.dropna()
#Stats Summary

data.describe().transpose()
#Function to show the details for specific year



levels = ["HIGH", "LOW"]

max_info = []

min_info = []

all_info = []



def details(x):

    df_details = pd.DataFrame(data[data['Year']==x])

    

    # -- Revenue Analysis --

    max_rev = df_details['Revenue'].max()

    min_rev = df_details['Revenue'].min()

    

    # -- Rating Analysis --

    max_rat = df_details['Rating'].max()

    min_rat = df_details['Rating'].min()

    

   

    for i, r in df_details.iterrows():

        if r['Revenue'] == max_rev:

            max_info.append(r['Title'])

        elif r['Revenue'] == min_rev:   

            min_info.append(r['Title'])

        elif r['Rating'] == max_rat:

            max_info.append(r['Title'])

        elif r['Rating'] == min_rat:   

            min_info.append(r['Title'])

            

    if len(max_info) > 2:

        max_info.pop(2)

    else:

        pass

        

    if len(min_info) > 2:

        min_info.pop(2)

    else:

        pass

        

    max_info.insert(0,"HIGH")

    min_info.insert(0,"LOW")

    

    all_info = [max_info, min_info]

    print(tabulate(all_info, headers=['', 'Revenue','Rating']))

    

    max_info.clear()

    min_info.clear()

    all_info.clear()

details(2015)
details(2016)