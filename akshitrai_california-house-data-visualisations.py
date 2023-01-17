import numpy as np 

import plotly.express as px

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/california-house-pricing/california_house.csv')
plt.figure(figsize=[15,1])



px.scatter(df,x='Longitude',y='Latitude',width=500,color='SalePrice',hover_data=['HouseAge'])
sns.heatmap(df.corr())
px.scatter(df,x='SalePrice',y='AveRooms',width=500)
px.scatter(df,x='SalePrice',y='AveBedrms',width=500)
print(f'Average House Age : ',df['HouseAge'].mean())
print(f'Average House Age with higher than 4 Crore Sale Price : ',df[df['SalePrice'] > 4]['HouseAge'].mean())
print(f'Average House Rooms : ',df['AveRooms'].mean())
print(f'Average House with Rooms higher than 4 Crore Sale Price  : ',df[df['SalePrice'] > 4]['AveRooms'].mean())
print(f'Average House BedRooms : ',df['AveBedrms'].mean())
print(f'Average House with Bedrooms higher than 4 Crore Sale Price  : ',df[df['SalePrice'] > 4]['AveBedrms'].mean())