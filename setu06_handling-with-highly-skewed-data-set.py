#importing the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading the dataset

data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
#look at dataset

data.head(2)
#Removing the unnamed: 0 column 

data = data.drop(['Unnamed: 0'], axis = 1)
data.head(2)
#Exploring the dataset information

data.info()
#checking dimension of dataset

data.shape
#importing library for visualising dataset and plotting the  histogram for  price attributes

import seaborn as sns

data['price'].hist(grid = False)
# Checking the skewness of Price column of dataset

data['price'].skew()
#density plot

sns.distplot(data['price'], hist = True)
# Checking the skewness of mileage column of dataset

data['mileage'].skew() 
sns.distplot(data['mileage'], hist = True)
#performing the log transformation using numpy

log_mileage = np.log(data['mileage'])

log_mileage
#checking the skewness after the log-transformation

log_mileage.skew()
#calculating the square root for data['mileage'] column

sqrt_mileage = np.sqrt(data['mileage'])

sqrt_mileage
#calculation skewness after calculating the square root 

sqrt_mileage.skew()
#visualising by density plot

sns.distplot(sqrt_mileage, hist = True)
#calculating the cube root for the column data['mileage'] column



cube_root_mileage = np.cbrt(data['mileage'])

cube_root_mileage
#calculation skewness after calculating the cube root 

cube_root_mileage.skew()
#visualising by density plot

sns.distplot(cube_root_mileage, hist = True)
#calculating the reciprocal for the column data['mileage'] column

recipr_mileage = np.reciprocal(data['mileage'])

recipr_mileage
recipr_mileage.skew()