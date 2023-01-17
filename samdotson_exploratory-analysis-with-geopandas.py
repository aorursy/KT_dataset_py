import os

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import geopandas as gp

import pyproj

import folium

from ipywidgets import widgets

from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual, Layout
Lincoln_blk_groups = gp.GeoDataFrame.from_file("../input/lincoln-block-groups/Lincoln_block_groups.geojson")

Lincoln_blk_groups.head()
labels=pd.read_csv("../input/metadata/defition.csv", sep=',')

print((labels.iloc[:,1]).head())
from sklearn.linear_model import LinearRegression

from ipywidgets import Button, Layout

b = Button(description='(50% width, 80px height) button',

           layout=Layout(width='50%', height='80px', margin="100px"))

def f(x='Median Age-Total Population', y='Median Value (Dollars)'):

    try:

        x_abbr = labels.loc[labels['label'] == x , 'abbreviated'].iloc[0]

        y_abbr = labels.loc[labels['label'] == y , 'abbreviated'].iloc[0] 

        x_values = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== x_abbr]).values)

        y_values = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== y_abbr]).values)

        regressor = LinearRegression()

        regressor.fit(x_values, y_values)

        fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')

        plt.scatter(x_values, y_values, color = 'red')

        plt.plot(x_values, regressor.predict(x_values), color = 'blue')

        plt.title(x + " versus " + y +" in Lincoln, NE", fontsize=25 )

        plt.xlabel(x , fontsize=15)

        plt.ylabel( y , fontsize=15)

        plt.show()

    except:  

        print("Opps, something went wrong! Perhaps you should try diffent fields")

        plt.close()

interact(f,x=labels.iloc[:,1], y=labels.iloc[:,1])

def f(x='Total Population', y='Total Population-Male'):

    try: 

        f, ax = plt.subplots(1, figsize=(15, 10))

        ax.set_title(x +" divided by " + y + ' in Lincoln, NE')

        denom = labels.loc[labels['label'] == x , 'abbreviated'].iloc[0]

        numer = labels.loc[labels['label'] == y , 'abbreviated'].iloc[0] 

        Lincoln_blk_groups[denom+" over "+numer] = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== denom]).values)/((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== numer ]).values)

        Lincoln_blk_groups.plot(denom+" over "+numer, scheme='fisher_jenks', k=5, cmap=plt.cm.Blues, legend=True, ax=ax)

        ax.set_axis_off()

        plt.axis('equal');

        plt.show()

        del Lincoln_blk_groups[denom+" over "+numer] 

    except: 

        print("Opps, something went wrong! Perhaps you should try diffent fields")

        plt.close()

interact(f,x=labels.iloc[:,1], y=labels.iloc[:,1])