import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
data_df = pd.read_csv(os.path.join("/kaggle", "input", "household-internet-connection-in-european-union", "isoc_ci_it_h.tsv"), sep='\t')
data_df.shape
data_df.head()
data_df.columns
data_df.columns = ['composed', '2019', '2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003']
data_df.head()
data_df['connection_type'] = data_df['composed'].apply(lambda x: x.split(",")[0])

data_df['unit'] = data_df['composed'].apply(lambda x: x.split(",")[1])

data_df['household_type'] = data_df['composed'].apply(lambda x: x.split(",")[2])

data_df['country'] = data_df['composed'].apply(lambda x: x.split(",")[3])
data_df.head()
data_df.country.unique()
data_df.connection_type.unique()
data_df.unit.unique()
data_df.household_type.unique()
a2_dch_data_df = data_df.loc[(data_df.household_type=="A2_DCH") & (data_df.connection_type=="H_BBFIX") & (data_df.unit=="PC_HH")]
a2_dch_data_df.shape
a2_dch_data_df.head(44)
import re

def clean_text(text):

    text = re.sub(r"[a-zA-Z: ]", "", text)

    text = text.replace(" ","")

    if text:

        text = int(text)

    else:

        text = None

    return text
a2_dch_data_df['2019'] = a2_dch_data_df['2019'].apply(lambda x: clean_text(x))
def plot_count(feature, value, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    df = df.sort_values([value], ascending=False).reset_index(drop=True)

    g = sns.barplot(df[feature][0:45], df[value], palette='Set3')

    g.set_title("Number of {}".format(title))

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    plt.show()    
plot_count("country", "2019", "Fixed internet connection / household with 2 adults and dependent children - 2019", a2_dch_data_df, size=4)
def plot_data_per_year(year):

    a2_dch_data_df[year] = a2_dch_data_df[year].apply(lambda x: clean_text(x))

    plot_count("country", f"{year}", f"Fixed internet connection / household with 2 adults and dependent children - {year}", a2_dch_data_df, size=4)        
plot_data_per_year('2018')
plot_data_per_year('2017')
plot_data_per_year('2016')
plot_data_per_year('2015')
plot_data_per_year('2014')
plot_data_per_year('2013')
plot_data_per_year('2012')
plot_data_per_year('2011')
plot_data_per_year('2010')
def plot_data_per_year_subset(hh_type='A2_DCH', con_type='H_BBFIX', unit='PC_HH', year='2019'):



    subset_df = data_df.loc[(data_df.household_type=="A2_DCH") & (data_df.connection_type=="H_BBFIX") & (data_df.unit=="PC_HH")]

    subset_df[year] = subset_df[year].apply(lambda x: clean_text(x))

    plot_count("country", f"{year}", f"internet connection: {con_type} household type: {hh_type} - year: {year}", subset_df, size=4)           
plot_data_per_year_subset(year='2019')
plot_data_per_year_subset(con_type='H_BBMOB', year='2019')
plot_data_per_year_subset(hh_type="A1", con_type='H_BBMOB', year='2019')
plot_data_per_year_subset(hh_type="A1", con_type='H_BBFIX', year='2019')