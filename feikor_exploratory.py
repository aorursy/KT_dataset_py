# Load basic packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

import matplotlib #collection of functions for scientific and publication-ready visualization

import numpy as np #foundational package for scientific computing

import scipy as sp #collection of functions for scientific computing and advance mathematics

import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

import sklearn #collection of machine learning algorithms

import pickle #saving and loading models

import os



# Visualization libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



# Common model helpers for preprocessing etc.

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Configure Visualization Defaults

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



inpdir = '../input'

print(os.listdir(inpdir))



# Any results you write to the current directory are saved as output.
def read_data(year):

    for dirs in os.listdir(inpdir):

        if year in dirs:

            d1 = inpdir + '/' + dirs

            n = os.listdir(d1)

            name = os.listdir(d1 + '/' + n[0] + '/data')

            return pd.read_csv(d1 + '/' + n[0] + '/data/' + name[0])

        

def trim_locations(df):

    location_columns = ['latitude', 'longitude']

    df.columns = [col.lower() for col in df.columns]

    for col in location_columns:

        df[col] = df[col].apply(pd.np.round(2))

    return df

        

# Used for basic data exploration, getting info, description,

# shape, amount of null values and some sample and head data

def basic_data_exploration(df):

    print('-'*20)

    print('Information about the dataset: ')

    print(df.info())

    print('-'*20)

    print('Data description: ')

    print(df.describe(include='all'))

    print('-'*20)

    print('Null values in our data: ')

    print(df.isnull().sum().sort_values(ascending=False))

    print('-'*20)

    print('Data shape: ')

    print(df.shape)

    print('-'*20)

    print('Data head (5): ')

    print(df.head(5))

    print('-'*20)

    print('Data sample (10): ')

    print(df.sample(10))

    print('-'*20)

    

# Encode categories, will give them integer values to process

def encode_categories(df, columns):

    

    for col in columns:

        df[col] = str(df[col])

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

    return df



        

# Read the data in, starting with 2010 - 2016 only, might extend

df2010 = read_data('2010')

df2011 = read_data('2011')

df2012 = read_data('2012')

df2013 = read_data('2013')

df2014 = read_data('2014')

df2015 = read_data('2015')

df2016 = read_data('2016')



dataframes = [df2010,df2011,df2012,df2013,df2014,df2015,df2016]

# Basic data exploration in numbers

#basic_data_exploration(df)
# Amount of crimes per year

year = 2010

x = []

y = []

for years in dataframes:

    x.append(year)

    year +=1

    y.append(years['delegacia'].count())



plt.bar(x, y, 0.8, align='center')

plt.ylabel('Amount of crimes')
# Amount of crimes per month in 2015

monthscrimes = df2015['mes'].value_counts()

months = monthscrimes.plot.bar(x='mes', y='count', rot=0)
# Amount of crimes per month in 2016

monthscrimes = df2016['mes'].value_counts()

months = monthscrimes.plot.bar(x='mes', y='count', rot=0)
# Amount of crimes per police station

policestationcrimes = df2016['delegacia'].value_counts().head(20)

policestations = policestationcrimes.plot.bar(x='delegacia', y='count', rot=90)

total = df2016['delegacia'].count()

print('Total amount of crimes in 2016 (with registered policstation): ' + str(total))

policestationsamount = len(df2016['delegacia'].unique())

print('Total numer of policestations registered: ' + str(policestationsamount))

print('Amount of crimes registered in the 20 biggest police stations: {:.2%} '.format((policestationcrimes.sum()/total)))
# Check amount of unique values for non-numeric features

numeric_features = df2016.select_dtypes(include=np.number).columns

non_numeric_features = df2016.select_dtypes(exclude=np.number).columns

for col in non_numeric_features:

    print (len(df2016[col].unique()))



# We could encode some of the features if we would want to model anything.

#

# df_enc = df2016.copy(deep=True)

# df_enc = encode_categories(df_enc, [col for col in non_numeric_features if len(df[col].unique())<1000])

# print(df_enc.info())

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

    

# Plot a simple correlation heatmap

correlation_heatmap(df2016)