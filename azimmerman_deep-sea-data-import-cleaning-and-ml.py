## Setup and Import



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy as np

print('np: {}'.format(np.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas as pd

print('pandas: {}'.format(pd.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))

import pickle

import random

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn import model_selection

#from sklearn.metrics import classification_report

#from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# %%Custom Functions

def flatten_base(iter):

    for object in iter:

        if hasattr(object,'__iter__') and not isinstance(object,str):

            yield from flatten(object)

        else:

            yield (object)

            

def flatten(input_): return list(flatten_base(input_))
#Import the Data from Github or NOAA's site directly (note NOAA import is much slower)

#default_data_url = "https://ecowatch.ncddc.noaa.gov/erddap/tabledap/deep_sea_corals.csv?Phylum%2CClass%2COrder%2CFamily%2CGenus%2CSpecies%2COcean%2CLargeMarineEcosystem%2Clatitude%2Clongitude%2CDepthInMeters%2CObservationYear"



ds_data_raw = pd.read_csv("../input/all-deep-sea-records2020feb14/all_deep_sea_records-2020-feb-14.csv")





# Check Dataframe Shape

 #print(data.shape)

 #print(data.ndim)
#print the header of the data to ensure it imported correctly

ds_data_raw.head()

print("Number of rows of data before NaN removed for entire world dataset: ",ds_data_raw.shape[0])
#save the headers as a list to use later

header = list(ds_data_raw.columns)

#drop anything that is NaN in important columns

dropNaN_columns = ['Genus','latitude','longitude','DepthInMeters','ObservationYear']

numeric_columns = ["latitude", "longitude","DepthInMeters","ObservationYear"]



#convert object datatypes to numeric where possible in needed columns, if not possible just make NaN

ds_data_tonumeric = ds_data_raw

ds_data_tonumeric[numeric_columns] = ds_data_raw[numeric_columns].apply(pd.to_numeric, errors='coerce')



ds_data = ds_data_tonumeric.dropna(subset=dropNaN_columns)

#print how many were dropped

print("Number of rows of data after NaN removed for entire world dataset: ",ds_data.shape[0])
#check to make sure data types are right

ds_data.dtypes
#filter out the data we want that NaN didn't catch (ie typoes in latitude or depths, etc)

year_low = 1960

year_high = 2020

depth_low = 50

depth_high = 2000

latitude_low = -90

latitude_high = 90

longitude_low = -180

longitude_high = 180 



#these need to be exactly the same string as the header row

numeric_filter_dict = { 

    'latrange': ('latitude',latitude_low,latitude_high),

    'longrange': ('longitude',longitude_low,longitude_high),

    'yearange': ('ObservationYear',year_low,year_high),

    'depthrange': ('DepthInMeters',depth_low,depth_high)

}



#filter using our low and high values

for _ in numeric_filter_dict:

    category_name = numeric_filter_dict[_][0] 

    ds_data = ds_data.loc[(ds_data[category_name] >= numeric_filter_dict[_][1]) & (ds_data[category_name] <= numeric_filter_dict[_][2])]

    

#print how many are left now

print("Number of rows of data after numeric filtering for entire world dataset: ",ds_data.shape[0])
#optional selections



#select only certain regions

regions_easternUS=('Gulf of Mexico','Southeast U.S. Continental Shelf') #,'Caribbean Sea'

ds_data_easternUS = ds_data.loc[ds_data['LargeMarineEcosystem'].isin(regions_easternUS)]



#select corals or sponges

orders_coral = ('Antipatharia', 'Scleractinia', 'Zoanthidea', 'Helioporacea', 'Alcyonacea', 'Pennatulacea', 'Anthoathecata','Gorgonacea')

phylum_coral = ('Cnidaria','Cnidaria') #->have to be list so that isin() works

phylum_sponge = ('Porifera','Porifera') #->have to be list so that isin() works

ds_data_corals = ds_data.loc[ds_data['Phylum'].isin(phylum_coral)]

ds_data_sponges = ds_data.loc[ds_data['Phylum'].isin(phylum_sponge)]



#example for how to do one line of code for complete selection 

ds_data_easternUS_corals = ds_data.loc[ds_data['LargeMarineEcosystem'].isin(regions_easternUS) & ds_data['Phylum'].isin(phylum_coral)]
##Now we are don with the data cleaning!##