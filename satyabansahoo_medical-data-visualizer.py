#Importing packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#Loading the dataset
data = pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv",sep=";")
data.head()
data.shape
data.describe()
data.info()
data.isnull().sum()
# Add 'overweight' column
data['overweight'] = np.where((data['weight']/((data['height']/100)**2)) > 25, 1, 0)
data.head()
# Normalize data by making 0 always good and 1 always bad. 
#If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
data['cholesterol'] = np.where(data['cholesterol']==1,0,1)
data['gluc'] = np.where(data['gluc']==1,0,1)
data.head()
# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` 
    #using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    data_cat = pd.melt(data,value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                       id_vars=['cardio'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    #You will have to rename one of the columns for the catplot to work correctly.
    data_cat = pd.DataFrame(data_cat.groupby(['variable', 'value', 'cardio'])['value']
                          .count()).rename(columns={'value': 'total'}).reset_index()    

    # Draw the catplot with 'sns.catplot()'
    fig = plt.figure(figsize=(8,8))
    sns.catplot(data = data_cat, x='variable', y='total', hue='value', col='cardio', kind="bar")

    # saving the png image
    fig.savefig('catplot.png')
    return fig
draw_cat_plot()
# Draw Heat Map
def draw_heat_map():
    # Clean the data
    data_heat = data[(data['ap_lo'] <= data['ap_hi']) & 
                    (data['height'] >= data['height'].quantile(0.025)) &
                    (data['height'] <= data['height'].quantile(0.975)) &
                    (data['weight'] >= data['weight'].quantile(0.025)) & 
                    (data['weight'] <= data['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = data_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11,9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=.3, center=0,
              square=True, linewidths=.9, cbar_kws={"shrink": .5})

    # saving the png image
    fig.savefig('heatmap.png')
draw_heat_map()