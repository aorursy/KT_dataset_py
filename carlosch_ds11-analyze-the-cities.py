# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # to basic visualization 

import seaborn as sns # to statictics visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load datasets

cities_data = pd.read_csv("../input/cities.csv")

cost_life_data = pd.read_csv("../input/movehubcostofliving.csv")

quality_life_data = pd.read_csv("../input/movehubqualityoflife.csv")
#show sample of datasets

cost_life_data.head()
#show sample of datasets

quality_life_data.head()
#show sample of data

cities_data.head()
#analyze if preliminary missing data in datasets

print("analyze of 'the cities' dataset: \n",cities_data.isnull().sum() )

print( "analyze of 'the cost of life' dataset: \n",cost_life_data.isnull().sum() )

print("analyze of 'the quality of life' dataset: \n",quality_life_data.isnull().sum() )
#show the dimension of datasets

print("dimension of row-columns of 'cities' data set:", cities_data.shape )

print("dimension of row-columns of 'cost of life' data set:", cost_life_data.shape )

print("dimension of row-columns of 'quality of life' data set:", quality_life_data.shape )
#see the row with missing data and fit the missing data in cities data set

missing_data = cities_data[ cities_data.isnull().any( axis = 1 ) ]

fit_missing_data = {'Sevastopol':'Rusia' , 'Simferopol':'Rusia' , 'Priština':'Kosovo' }

for  first,second in fit_missing_data.items() :

    cities_data.ix[ cities_data.City == first, 'Country'] = second

cities_data.isnull().sum()
#integrate/merge datasets to the unique dataset and cleaning row with missing data

data_proof = pd.merge( cost_life_data , quality_life_data, how = 'right', on = "City" )

data_proof['City'] = data_proof['City'].str.lower()

cities_data['City'] = cities_data['City'].str.lower()

#convert series to list to join with data_proof by selected cities without missing data

cities_selected = cities_data[ cities_data['City'].isin( data_proof['City'] ) ]

data_proof = pd.merge(  data_proof , cities_selected , how = 'right', on = 'City' )

#show if dataset have missing value in any columns

data_proof.head()

print( data_proof.isnull().sum(),'\n',data_proof.shape )

#get the number of country and sort it

country = data_proof.groupby(["Country"]).size().sort_values(ascending =  False).to_frame()

country.columns = ["Size"]

country