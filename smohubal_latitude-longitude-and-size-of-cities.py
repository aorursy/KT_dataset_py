import os

import re

import seaborn           as sns

import matplotlib.pyplot as plt

import pandas            as pd

import numpy             as np



import warnings

warnings.filterwarnings('ignore')
class analysis:

    

    def __init__(self, data):

        self.data   = data

    

    def missing_percent_plot(self):

        missing_col = list(self.data.isna().sum() != 0)



        try:

            if True not in missing_col:

                raise ValueError("There is no missing values.")



            self.data = self.data.loc[:,missing_col]

            missing_percent = (self.data.isna().sum()/ self.data.shape[0]) * 100



            df = pd.DataFrame()

            df['Total']        = self.data.isna().sum()

            df['perc_missing'] = missing_percent

            p = sns.barplot(x=df.perc_missing.index, y='perc_missing', data=df); plt.xticks(rotation=90)

            plt.xticks(rotation=45);p.tick_params(labelsize=14)

        except:

            return print('There is no missing values...')

        return df.sort_values(ascending =False, by='Total', axis =0)

    

    def plots(self, columns : list, hue_col = None, sort = 'single'):

        _, axs = plt.subplots(int(round(len(columns) / 2, 0)), 5,figsize=(12,12))

        

        if sort == 'hue':

            for n, c in enumerate(columns):

                # hue loop

                for hue_value in self.data[hue_col].unique():

                    sns.distplot(self.data[self.data[hue_col] == hue_value][c], hist = False, label=hue_value, ax=axs[n//5][n%5])

                plt.tight_layout()

            plt.show()

        

        elif sort == 'single':

            for n, c in enumerate(columns):

                sns.distplot(self.data[c], hist = False, ax=axs[n//5][n%5])

                plt.tight_layout()

            plt.show()
new_geo = pd.read_csv('../input/162cities_info.csv', index_col=0)

vis_df = new_geo[~new_geo['total(m^2)'].isna() & ~new_geo['population'].isna() & ~new_geo['formatted_addres_name'].isna()]

vis_df.shape
vis_df.sample(1)
pp = analysis(vis_df.fillna(0).loc[:, list(vis_df.select_dtypes('number').columns)].apply(np.log))

pp.plots(list(vis_df.select_dtypes('number').columns))
vis_df.sample(1)
vis_df.reset_index(drop=True, inplace=True)
def feature_correlation_heatmap(importances_df, start=0, last=10):

    def heatmap(features):    

        sns.set(style="white")



        # Compute the correlation matrix

        corr = importances_df.loc[:,features].corr()



        # Generate a mask for the upper triangle

        mask = np.zeros_like(corr, dtype=np.bool)

        mask[np.triu_indices_from(mask)] = True



        # Set up the matplotlib figure

        f, ax = plt.subplots(figsize=(11, 9))



        # Generate a custom diverging colormap

        cmap = sns.diverging_palette(220, 10, as_cmap=True)



        # Draw the heatmap with the mask and correct aspect ratio

        return sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

                           square=True, linewidths=.5, cbar_kws={"shrink": .5}, 

                           annot =True, annot_kws = {'size':9})



    n_to_n = list(importances_df.columns)[start:last]

    return heatmap(n_to_n)
feature_correlation_heatmap(vis_df.loc[:, ['population', 'city(m^2)', 'geo_location_lat', 'geo_location_lng']])
num_vis_df = vis_df.select_dtypes('number').columns
for c in num_vis_df:

    num_vis_df[c] = num_vis_df[c].apply(np.log)
vis_df.head()
vis_df01 = vis_df.copy()
for city in vis_df01.address_compo_short_name.value_counts()[vis_df01.address_compo_short_name.value_counts().values < 3].index:

    vis_df01.address_compo_short_name[vis_df01.address_compo_short_name == city] = 'primary_cities'
for c in num_vis_df:

    vis_df01[c] = vis_df01[c].apply(np.log)
vis_df01.area.value_counts()
for city in vis_df01.address_compo_short_name.value_counts()[vis_df01.address_compo_short_name.value_counts().values < 3].index:

    vis_df01.address_compo_short_name[vis_df01.address_compo_short_name == city] = 'primary_cities'
vis_df01.area[vis_df01.area == '영남권'] = 'Yeongnam'

vis_df01.area[vis_df01.area == '호남권'] = 'Honam'

vis_df01.area[vis_df01.area == '수도권'] = 'Metropolitan'



vis_df01.area[vis_df01.area == '충청권'] = 'Chungcheong'

vis_df01.area[vis_df01.area == '강원권'] = 'Gangwon'

vis_df01.area[vis_df01.area == '제주권'] = 'Jeju'
vis_df01.head()
sns.scatterplot(x = 'geo_location_lng',y = 'total(m^2)', hue= 'area', data =vis_df01 )
sns.pairplot(vis_df01.loc[:, ['total(m^2)', 'geo_location_lat', 'geo_location_lng', 'population', 'area']], hue= 'area')
import plotly.graph_objs as go

import plotly            as py

from plotly.offline      import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON NOTEBOOK WHILE KERNEL IS RUNNING



data = []

clusters = []

colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)', 'rgb(255,175,255)', 'rgb(0,0,0)', 'rgb(0,255,255)']



for i in range(len(vis_df01.area.unique())):

    name = vis_df01.area.unique()[i]

    color = colors[i]

    x = vis_df01[ vis_df01['area'] == name ]['geo_location_lat']

    y = vis_df01[ vis_df01['area'] == name ]['geo_location_lng']

    z = vis_df01[ vis_df01['area'] == name ]['total(m^2)']

    

    trace = dict(

        name = name,

        x = x, y = y, z = z,

        type = "scatter3d",    

        mode = 'markers',

        marker = dict(size=5, color=color, line=dict(width=0)))

    data.append(trace)

    

    cluster = dict(

        color = color,

        opacity = 0.3,

        type = "mesh3d",    

        x = x, y = y, z = z )

    data.append(cluster)





    

layout = dict(

    width=800,

    height=550,

    autosize=False,

    title=dict(text = 'Correlation between location and size of cities   ',

               font = dict(size=20, family = 'Old Standard TT')),

    

    # scene used for 3d axes

    scene=dict(

        xaxis=dict(

            gridcolor='rgb(255, 255, 255)',

            zerolinecolor='rgb(255, 255, 255)',

            showbackground=True,

            backgroundcolor='rgb(230, 230,230)',

            title=dict(text = 'Latitude',

                       font = dict(size=14))

            

        ),

        yaxis=dict(

            gridcolor='rgb(255, 255, 255)',

            zerolinecolor='rgb(255, 255, 255)',

            showbackground=True,

            backgroundcolor='rgb(230, 230,230)',

            title=dict(text = 'Longitude',

                       font = dict(size=14))

        ),

        zaxis=dict(

            gridcolor='rgb(255, 255, 255)',

            zerolinecolor='rgb(255, 255, 255)',

            showbackground=True,

            backgroundcolor='rgb(230, 230,230)',

            title=dict(text = 'Size of cities',

                       font = dict(size=14))

        ),

        aspectratio = dict( x=1, y=1, z=0.7 ),

        aspectmode = 'manual'),)



fig = dict(data=data, layout=layout)

iplot(fig)