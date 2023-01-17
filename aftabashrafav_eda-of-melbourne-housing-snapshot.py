import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.gridspec as gridspec

from matplotlib.gridspec import GridSpec



import seaborn as sns



import missingno as msno 



#Plotly

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



#Some styling

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")



import plotly.io as pio

pio.templates.default = "gridon"



#Subplots

from plotly.subplots import make_subplots



#Showing full path of datasets

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Disable warnings 

import warnings

warnings.filterwarnings('ignore')

#reading the dataframe

melbourne_data=pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')

#First 5 rows of our dataset



melbourne_data.head()

#Number of rows and columns

melbourne_data.shape

#Columns in our dataset

melbourne_data.columns.unique()
#Description of our dataset

melbourne_data.describe().T



#T refers to transpose that displays the description of our dataset in long format.

melbourne_data.head()
#summary of a dataframe

melbourne_data.info()
melbourne_data.dtypes.value_counts()

#it will give the different types of data type present in the data frame and its count




#Let us examine numerical features in the train dataset

num_col=melbourne_data.select_dtypes(exclude='object')





#Let us examine categorical features in the train dataset

cat_col=melbourne_data.select_dtypes(exclude=['int64','float64'])
num_col.isnull().sum()

# let's try to visualize the number of missing values in each numerical feature.
cat_col.isnull().sum()

# let's try to visualize the number of missing values in each categorical feature.
num_col.head()
msno.heatmap(num_col);

#it will give the heatmap of missing value of numerical feature
#correlation map

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[8:9,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.kdeplot(num_col.BuildingArea,Label='BuildingArea',color='b');
#correlation map

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[9:10,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.kdeplot(num_col.YearBuilt,Label='YearBuilt',color='b');
#correlation map

f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[6:7,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.kdeplot(num_col.Car,Label='Car',color='b');
melbourne_data.groupby(['Rooms'])['Car'].mean()
melbourne_data.mode(axis=0)

#Get the mode(s) of each element along the selected axis.

#The mode of a set of values is the value that appears most often. It can be multiple values.
#Replaced the nan value in the car with mode value ie,2

melbourne_data.Car.replace({np.nan:2},inplace=True)

#Replaced the nan value in the BuildingArea with mean

melbourne_data['BuildingArea'].replace({np.nan:melbourne_data['BuildingArea'].mean()},inplace=True)
melbourne_data['CouncilArea']=melbourne_data['CouncilArea'].replace(np.NaN,0)
#Replaced the nan value in the car with YearBuilt

melbourne_data['YearBuilt'].replace({np.NaN:melbourne_data['YearBuilt'].mean()},inplace=True)
melbourne_data.isna().sum()
melbourne_data['Price'].describe()
plt.figure(figsize=(16,6))

sns.lineplot(data=melbourne_data['Price'])

plt.title('Price')
#Creating a heat map of all the numerical features.

plt.figure(figsize=(10,10))

mat = np.round(num_col.corr(), decimals=2)

sns.heatmap(data=mat, linewidths=1, linecolor='black');




#Getting features that have a correlation value greater than 0.5 against sale price.

for val in range(len(mat['Price'])):

    if abs(mat['Price'].iloc[val]) > 0.3:

        print(mat['Price'].iloc[val:val+1]) 





melbourne_data.groupby(['Regionname','CouncilArea'])['Price'].mean()
#analysis by Regional features

region_feature=['Regionname','CouncilArea']

plt.figure(figsize=(16,40))

i=1

for feature in region_feature:

    plt.subplot(3,1,i)

    sns.boxplot(y=melbourne_data[feature],x=melbourne_data['Price'],palette="bright")

    plt.ylabel(feature)

    i+=1

#we can analyse the price and type by region wise

plt.figure(figsize=(10,10))

sns.barplot(x=melbourne_data['Price'],y=melbourne_data['Regionname'],hue=melbourne_data['Type'],palette='dark');
#we can analyse the price and Rooms by region wise

plt.figure(figsize=(10,10))

sns.barplot(x=melbourne_data['Price'],y=melbourne_data['Regionname'],hue=melbourne_data['Rooms'],palette='bright');
#we can analyse the price and Rooms by region wise

plt.figure(figsize=(10,10))

sns.barplot(x=melbourne_data['Price'],y=melbourne_data['Type'],hue=melbourne_data['Rooms'],palette='deep');
melbourne_data.groupby(['Rooms','Bathroom'])['Price'].mean()
#analysis by roomfeatures

room_feature=['Bedroom2','Bathroom','Rooms']

plt.figure(figsize=(10,20))

i=1

for feature in room_feature:

    plt.subplot(3,1,i)

    sns.boxplot(x=melbourne_data[feature],y=melbourne_data['Price'])

    plt.xlabel(feature)

    i+=1
plt.title('Relation between Rooms,Bathroom and Price')

sns.scatterplot(y=melbourne_data['Bathroom'],x=melbourne_data['Rooms'],hue=melbourne_data['Price']);